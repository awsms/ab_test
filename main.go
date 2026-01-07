package main

import (
	"bytes"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"os/exec"
	"sync"

	"golang.org/x/term"

	"github.com/faiface/beep"
	"github.com/faiface/beep/speaker"
	"github.com/faiface/beep/wav"
)

type Switcher struct {
	mu      sync.RWMutex
	buffers []*beep.Buffer
	cur     int // current buffer index
	pos     int // current sample index (frames)
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
// It streams from the currently selected buffer starting at s.pos.
// Switching files keeps s.pos constant => sample-aligned switching.
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

func ffmpegWavDecode(path string) (beep.StreamSeekCloser, beep.Format, func() error, error) {
	// Decode *any* audio to WAV on stdout, preserving SR/channels by default.
	cmd := exec.Command(
		"ffmpeg",
		"-hide_banner",
		"-loglevel", "error",
		"-i", path,
		"-vn",
		"-f", "wav",
		"pipe:1",
	)

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, beep.Format{}, nil, err
	}
	var stderr bytes.Buffer
	cmd.Stderr = &stderr

	if err := cmd.Start(); err != nil {
		return nil, beep.Format{}, nil, err
	}

	// wav.Decode expects an io.ReadCloser; stdout already is one.
	stream, format, err := wav.Decode(struct {
		io.Reader
		io.Closer
	}{Reader: stdout, Closer: stdout})
	if err != nil {
		_ = cmd.Wait()
		return nil, beep.Format{}, nil, fmt.Errorf("wav decode (via ffmpeg) failed: %v; ffmpeg: %s", err, stderr.String())
	}

	wait := func() error {
		// Close streamer first, then wait for ffmpeg.
		_ = stream.Close()
		if err := cmd.Wait(); err != nil {
			return fmt.Errorf("ffmpeg failed: %v; ffmpeg: %s", err, stderr.String())
		}
		return nil
	}

	return stream, format, wait, nil
}

func decodeToBufferFFmpeg(path string, want *beep.Format) (*beep.Buffer, beep.Format, error) {
	stream, format, wait, err := ffmpegWavDecode(path)
	if err != nil {
		return nil, beep.Format{}, err
	}
	defer func() {
		_ = wait()
	}()

	if want != nil {
		if format.SampleRate != want.SampleRate || format.NumChannels != want.NumChannels {
			return nil, beep.Format{}, fmt.Errorf(
				"format mismatch for %s: got %d Hz, %d ch; want %d Hz, %d ch",
				path, format.SampleRate, format.NumChannels, want.SampleRate, want.NumChannels,
			)
		}
	}

	buf := beep.NewBuffer(format)
	buf.Append(stream) // decode whole file into memory
	return buf, format, nil
}

func main() {
	var startIndex int
	var showFilename bool
	flag.IntVar(&startIndex, "i", 0, "start file index")
	flag.BoolVar(&showFilename, "show-filename", false, "show filenames while switching (NOT blind)")
	flag.Parse()
	paths := flag.Args()

	if len(paths) < 2 {
		fmt.Fprintf(os.Stderr, "Usage: %s [-i startIndex] [--show-filename] file1 file2 [file3...]\n", os.Args[0])
		os.Exit(2)
	}

	// First file establishes output format (decoded via ffmpeg->wav).
	firstBuf, format, err := decodeToBufferFFmpeg(paths[0], nil)
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
	}

	// Load remaining files
	for _, p := range paths[1:] {
		buf, _, err := decodeToBufferFFmpeg(p, &format)
		if err != nil {
			log.Fatalf("decode %s: %v", p, err)
		}
		switcher.buffers = append(switcher.buffers, buf)
	}

	if startIndex < 0 || startIndex >= len(switcher.buffers) {
		startIndex = 0
	}
	switcher.cur = startIndex

	fmt.Println("Controls: ←/→ switch, q quit")
	fmt.Printf("Loaded %d files. Format: %d Hz, %d ch\n", len(paths), format.SampleRate, format.NumChannels)
	if showFilename {
		fmt.Printf("Start: [%d] %s\n", switcher.cur, paths[switcher.cur])
	}

	// Start playback
	speaker.Play(switcher)

	// Raw keyboard input
	oldState, err := term.MakeRaw(int(os.Stdin.Fd()))
	if err != nil {
		log.Fatalf("term raw mode: %v", err)
	}
	defer term.Restore(int(os.Stdin.Fd()), oldState)

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

		// Arrow keys: ESC [ D (left), ESC [ C (right)
		if n >= 3 && buf[0] == 0x1b && buf[1] == '[' {
			switch buf[2] {
			case 'D': // left
				speaker.Lock()
				switcher.Add(-1)
				cur := switcher.cur
				speaker.Unlock()

				if showFilename {
					fmt.Printf("\rNow: [%d] %s            ", cur, paths[cur])
				}
			case 'C': // right
				speaker.Lock()
				switcher.Add(+1)
				cur := switcher.cur
				speaker.Unlock()

				if showFilename {
					fmt.Printf("\rNow: [%d] %s            ", cur, paths[cur])
				}
			}
		}
	}
}
