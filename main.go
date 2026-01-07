package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"sync"

	"golang.org/x/term"

	"github.com/faiface/beep"
	"github.com/faiface/beep/flac"
	"github.com/faiface/beep/mp3"
	"github.com/faiface/beep/speaker"
	"github.com/faiface/beep/vorbis"
	"github.com/faiface/beep/wav"
)

type Switcher struct {
	mu      sync.RWMutex
	buffers []*beep.Buffer
	cur     int // current buffer index
	pos     int // current sample index (frames)
	format  beep.Format
}

func (s *Switcher) SetIndex(i int) {
	s.mu.Lock()
	s.cur = i
	s.mu.Unlock()
}

func (s *Switcher) Add(delta int) {
	s.mu.Lock()
	n := len(s.buffers)
	s.cur = (s.cur + delta) % n
	if s.cur < 0 {
		s.cur += n
	}
	s.mu.Unlock()
}

// Stream implements beep.Streamer.
// It always streams from the currently selected buffer starting at s.pos.
// Switching files keeps s.pos constant, so playback stays sample-aligned.
func (s *Switcher) Stream(samples [][2]float64) (n int, ok bool) {
	s.mu.RLock()
	buf := s.buffers[s.cur]
	pos := s.pos
	s.mu.RUnlock()

	streamer := buf.Streamer(pos, pos+len(samples))
	n, ok = streamer.Stream(samples)

	s.mu.Lock()
	s.pos += n
	s.mu.Unlock()

	return n, ok
}

func (s *Switcher) Err() error { return nil }

func decodeToBuffer(path string, want beep.Format) (*beep.Buffer, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var (
		stream beep.StreamSeekCloser
		format beep.Format
	)
	ext := fileExtLower(path)

	switch ext {
	case ".wav":
		stream, format, err = wav.Decode(f)
	case ".mp3":
		stream, format, err = mp3.Decode(f)
	case ".flac":
		stream, format, err = flac.Decode(f)
	case ".ogg":
		stream, format, err = vorbis.Decode(f)
	default:
		return nil, fmt.Errorf("unsupported extension %q (supported: .wav .mp3 .flac .ogg)", ext)
	}
	if err != nil {
		return nil, err
	}
	defer stream.Close()

	// Since you assume sample-synced, we enforce identical format.
	if format.SampleRate != want.SampleRate || format.NumChannels != want.NumChannels {
		return nil, fmt.Errorf(
			"format mismatch for %s: got %d Hz, %d ch; want %d Hz, %d ch",
			path, format.SampleRate, format.NumChannels, want.SampleRate, want.NumChannels,
		)
	}

	buf := beep.NewBuffer(want)
	buf.Append(stream) // decodes the whole file into memory
	return buf, nil
}

func decodeFirst(path string) (*beep.Buffer, beep.Format, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, beep.Format{}, err
	}
	defer f.Close()

	var (
		stream beep.StreamSeekCloser
		format beep.Format
	)
	ext := fileExtLower(path)

	switch ext {
	case ".wav":
		stream, format, err = wav.Decode(f)
	case ".mp3":
		stream, format, err = mp3.Decode(f)
	case ".flac":
		stream, format, err = flac.Decode(f)
	case ".ogg":
		stream, format, err = vorbis.Decode(f)
	default:
		return nil, beep.Format{}, fmt.Errorf("unsupported extension %q (supported: .wav .mp3 .flac .ogg)", ext)
	}
	if err != nil {
		return nil, beep.Format{}, err
	}
	defer stream.Close()

	buf := beep.NewBuffer(format)
	buf.Append(stream)
	return buf, format, nil
}

func fileExtLower(path string) string {
	// minimal dependency: find last dot
	lastDot := -1
	for i := len(path) - 1; i >= 0; i-- {
		if path[i] == '.' {
			lastDot = i
			break
		}
		if path[i] == '/' || path[i] == '\\' {
			break
		}
	}
	if lastDot == -1 {
		return ""
	}
	ext := path[lastDot:]
	// ASCII lower
	b := []byte(ext)
	for i := range b {
		if b[i] >= 'A' && b[i] <= 'Z' {
			b[i] = b[i] - 'A' + 'a'
		}
	}
	return string(b)
}

func main() {
	var startIndex int
	flag.IntVar(&startIndex, "i", 0, "start file index")
	flag.Parse()
	paths := flag.Args()

	if len(paths) < 2 {
		fmt.Fprintf(os.Stderr, "Usage: %s [-i startIndex] file1 file2 [file3...]\n", os.Args[0])
		os.Exit(2)
	}

	// Load first file to establish format
	firstBuf, format, err := decodeFirst(paths[0])
	if err != nil {
		log.Fatalf("decode %s: %v", paths[0], err)
	}

	// Init speaker
	bufferSize := format.SampleRate.N(50e6) // ~50ms
	if bufferSize < 1024 {
		bufferSize = 1024
	}
	if err := speaker.Init(format.SampleRate, bufferSize); err != nil {
		log.Fatalf("speaker init: %v", err)
	}

	switcher := &Switcher{
		buffers: []*beep.Buffer{firstBuf},
		cur:     0,
		pos:     0,
		format:  format,
	}

	// Load remaining files
	for _, p := range paths[1:] {
		buf, err := decodeToBuffer(p, format)
		if err != nil {
			log.Fatalf("decode %s: %v", p, err)
		}
		switcher.buffers = append(switcher.buffers, buf)
	}

	if startIndex < 0 || startIndex >= len(switcher.buffers) {
		startIndex = 0
	}
	switcher.cur = startIndex

	fmt.Println("Controls: ←/→ switch file, q quit")
	fmt.Printf("Loaded %d files. Format: %d Hz, %d ch\n", len(paths), format.SampleRate, format.NumChannels)
	fmt.Printf("Start: [%d] %s\n", switcher.cur, paths[switcher.cur])

	// Start playback
	speaker.Play(switcher)

	// Raw keyboard input
	oldState, err := term.MakeRaw(int(os.Stdin.Fd()))
	if err != nil {
		log.Fatalf("term raw mode: %v", err)
	}
	defer term.Restore(int(os.Stdin.Fd()), oldState)

	// Read keypresses
	buf := make([]byte, 16)
	for {
		n, err := os.Stdin.Read(buf)
		if err != nil {
			log.Fatalf("stdin read: %v", err)
		}
		if n == 0 {
			continue
		}

		// q to quit
		if n == 1 && (buf[0] == 'q' || buf[0] == 'Q') {
			fmt.Println("\nBye.")
			return
		}

		// Arrow keys are usually ESC [ D (left) and ESC [ C (right)
		if n >= 3 && buf[0] == 0x1b && buf[1] == '[' {
			switch buf[2] {
			case 'D': // left
				speaker.Lock()
				switcher.Add(-1)
				cur := switcher.cur
				speaker.Unlock()
				fmt.Printf("\rNow: [%d] %s            ", cur, paths[cur])
			case 'C': // right
				speaker.Lock()
				switcher.Add(+1)
				cur := switcher.cur
				speaker.Unlock()
				fmt.Printf("\rNow: [%d] %s            ", cur, paths[cur])
			}
		}
	}
}
