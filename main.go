package main

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"sync"

	"golang.org/x/term"

	"github.com/faiface/beep"
	"github.com/faiface/beep/speaker"
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

func (s *Switcher) Stream(samples [][2]float64) (n int, ok bool) {
	s.mu.RLock()
	buf := s.buffers[s.cur]
	pos := s.pos
	s.mu.RUnlock()

	bufLen := buf.Len()
	if bufLen <= 0 || pos >= bufLen {
		for i := range samples {
			samples[i][0], samples[i][1] = 0, 0
		}
		return 0, false
	}

	end := pos + len(samples)
	if end > bufLen {
		end = bufLen
	}

	streamer := buf.Streamer(pos, end)
	n, ok = streamer.Stream(samples[:end-pos])

	// pad remainder if EOF
	for i := n; i < len(samples); i++ {
		samples[i][0], samples[i][1] = 0, 0
	}

	s.mu.Lock()
	s.pos += n
	s.mu.Unlock()

	return n, ok
}

func (s *Switcher) Err() error { return nil }

type ffprobeOut struct {
	Streams []struct {
		SampleRate string `json:"sample_rate"`
		Channels   int    `json:"channels"`
	} `json:"streams"`
}

func runFFprobe(path string, verbose bool) (sr int, ch int, rawJSON string, err error) {
	// ffprobe -v error -select_streams a:0 -show_entries stream=sample_rate,channels -of json <file>
	cmd := exec.Command(
		"ffprobe",
		"-v", "error",
		"-select_streams", "a:0",
		"-show_entries", "stream=sample_rate,channels",
		"-of", "json",
		path,
	)
	out, e := cmd.Output()
	if e != nil {
		return 0, 0, "", fmt.Errorf("ffprobe failed for %s: %v", path, e)
	}
	rawJSON = string(out)
	if verbose {
		fmt.Fprintf(os.Stderr, "ffprobe json for %s:\n%s\n", path, rawJSON)
	}

	var p ffprobeOut
	if e := json.Unmarshal(out, &p); e != nil {
		return 0, 0, rawJSON, fmt.Errorf("ffprobe json parse failed: %v", e)
	}
	if len(p.Streams) == 0 {
		return 0, 0, rawJSON, fmt.Errorf("ffprobe: no audio streams found in %s", path)
	}

	ch = p.Streams[0].Channels
	if ch <= 0 {
		return 0, 0, rawJSON, fmt.Errorf("ffprobe: invalid channels=%d for %s", ch, path)
	}

	sr64, e := strconv.ParseInt(strings.TrimSpace(p.Streams[0].SampleRate), 10, 32)
	if e != nil || sr64 <= 0 {
		return 0, 0, rawJSON, fmt.Errorf("ffprobe: invalid sample_rate=%q for %s", p.Streams[0].SampleRate, path)
	}
	sr = int(sr64)

	return sr, ch, rawJSON, nil
}

func shellish(args []string) string {
	var b strings.Builder
	for i, a := range args {
		if i > 0 {
			b.WriteByte(' ')
		}
		if a == "" || strings.ContainsAny(a, " \t\n\"") {
			b.WriteByte('"')
			b.WriteString(strings.ReplaceAll(a, `"`, `\"`))
			b.WriteByte('"')
		} else {
			b.WriteString(a)
		}
	}
	return b.String()
}

// decodeToBufferFFmpegRawFloat32 decodes any audio file using ffmpeg to raw f32le PCM.
// IMPORTANT about "avoid resampling":
//   - We DO NOT pass -ar (sample rate) or -ac (channels). So ffmpeg outputs the original
//     stream's sample rate & channel count (no resample/downmix).
//   - We query sr/ch via ffprobe so we can interpret the raw bytes correctly.
func decodeToBufferFFmpegRawFloat32(path string, want *beep.Format, verbose bool) (*beep.Buffer, beep.Format, error) {
	sr, ch, _, err := runFFprobe(path, verbose)
	if err != nil {
		return nil, beep.Format{}, err
	}

	if ch == 1 {
		// ok (we'll duplicate to stereo)
	} else if ch == 2 {
		// ok
	} else {
		return nil, beep.Format{}, fmt.Errorf("unsupported channel count %d for %s (only 1 or 2 supported)", ch, path)
	}

	args := []string{
		"-hide_banner",
		"-loglevel", "error",
		"-i", path,
		"-vn",
		"-map", "0:a:0",
		// No -ar, no -ac => no resampling / no channel remix.
		"-f", "f32le",
		"-c:a", "pcm_f32le",
		"pipe:1",
	}
	cmd := exec.Command("ffmpeg", args...)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr

	if verbose {
		fmt.Fprintf(os.Stderr, "cmd: ffmpeg %s\n", shellish(args))
	}

	raw, err := cmd.Output()
	if err != nil {
		return nil, beep.Format{}, fmt.Errorf("ffmpeg failed for %s: %v; ffmpeg: %s", path, err, stderr.String())
	}
	if len(raw) == 0 {
		return nil, beep.Format{}, fmt.Errorf("ffmpeg produced 0 bytes for %s; ffmpeg: %s", path, stderr.String())
	}
	if len(raw)%4 != 0 {
		return nil, beep.Format{}, fmt.Errorf("ffmpeg produced %d bytes (not multiple of 4) for %s", len(raw), path)
	}

	// Convert float32 little-endian samples to beep stereo frames.
	// If mono, duplicate to L/R. If stereo, map as-is.
	totalF32 := len(raw) / 4
	if totalF32%ch != 0 {
		return nil, beep.Format{}, fmt.Errorf("raw sample count %d not divisible by channels %d for %s", totalF32, ch, path)
	}
	frames := totalF32 / ch
	if verbose {
		fmt.Fprintf(os.Stderr, "decoded: sr=%d ch=%d float32-samples=%d frames=%d\n", sr, ch, totalF32, frames)
	}

	// Build a streamer over the frames without huge intermediate conversions.
	// We'll still ultimately store in beep.Buffer (RAM), consistent with your “instant switch” requirement.
	format := beep.Format{
		SampleRate:  beep.SampleRate(sr),
		NumChannels: 2, // speaker wants stereo frames [2]float64; beep uses stereo sample arrays
		Precision:   2, // not super meaningful here; keep 2 like typical 16-bit
	}

	// If enforcing same sr/ch across files:
	if want != nil {
		if format.SampleRate != want.SampleRate || format.NumChannels != want.NumChannels {
			return nil, beep.Format{}, fmt.Errorf(
				"format mismatch for %s: got %d Hz, %d ch; want %d Hz, %d ch",
				path, format.SampleRate, format.NumChannels, want.SampleRate, want.NumChannels,
			)
		}
	}

	// Streamer that reads from raw bytes and emits stereo float64 frames.
	var idx int // float32 index (not byte index)
	streamer := beep.StreamerFunc(func(out [][2]float64) (n int, ok bool) {
		// remaining frames in raw
		remainingFrames := frames - (idx / ch)
		if remainingFrames <= 0 {
			return 0, false
		}
		max := len(out)
		if remainingFrames < max {
			max = remainingFrames
		}

		for i := 0; i < max; i++ {
			if ch == 1 {
				v := f32From(raw, idx)
				idx += 1
				out[i][0] = clampFloat64(float64(v))
				out[i][1] = clampFloat64(float64(v))
			} else {
				l := f32From(raw, idx)
				r := f32From(raw, idx+1)
				idx += 2
				out[i][0] = clampFloat64(float64(l))
				out[i][1] = clampFloat64(float64(r))
			}
		}
		return max, true
	})

	buf := beep.NewBuffer(format)
	buf.Append(streamer)
	if buf.Len() == 0 {
		return nil, beep.Format{}, fmt.Errorf("decoded 0 frames from %s (unexpected)", path)
	}

	if verbose {
		fmt.Fprintf(os.Stderr, "buffer frames stored: %d\n", buf.Len())
	}
	return buf, format, nil
}

func f32From(b []byte, f32Index int) float32 {
	off := f32Index * 4
	u := binary.LittleEndian.Uint32(b[off : off+4])
	return math.Float32frombits(u)
}

func clampFloat64(x float64) float64 {
	if x > 1.0 {
		return 1.0
	}
	if x < -1.0 {
		return -1.0
	}
	if math.IsNaN(x) || math.IsInf(x, 0) {
		return 0.0
	}
	return x
}

func main() {
	var startIndex int
	var showFilename bool
	var verbose bool
	flag.IntVar(&startIndex, "i", 0, "start file index")
	flag.BoolVar(&showFilename, "show-filename", false, "show filenames while switching (NOT blind)")
	flag.BoolVar(&verbose, "verbose", false, "print ffprobe/ffmpeg diagnostics to stderr")
	flag.Parse()
	paths := flag.Args()

	if len(paths) < 2 {
		fmt.Fprintf(os.Stderr, "Usage: %s [-i startIndex] [--show-filename] [--verbose] file1 file2 [file3...]\n", os.Args[0])
		os.Exit(2)
	}

	firstBuf, format, err := decodeToBufferFFmpegRawFloat32(paths[0], nil, verbose)
	if err != nil {
		log.Fatalf("decode %s: %v", paths[0], err)
	}

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

	for _, p := range paths[1:] {
		buf, _, err := decodeToBufferFFmpegRawFloat32(p, &format, verbose)
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

	speaker.Play(switcher)

	oldState, err := term.MakeRaw(int(os.Stdin.Fd()))
	if err != nil {
		log.Fatalf("term raw mode: %v", err)
	}
	defer term.Restore(int(os.Stdin.Fd()), oldState)

	keybuf := make([]byte, 16)
	for {
		n, err := os.Stdin.Read(keybuf)
		if err != nil {
			log.Fatalf("stdin read: %v", err)
		}
		if n == 0 {
			continue
		}
		if n == 1 && (keybuf[0] == 'q' || keybuf[0] == 'Q') {
			fmt.Println("\nBye.")
			return
		}
		if n >= 3 && keybuf[0] == 0x1b && keybuf[1] == '[' {
			switch keybuf[2] {
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
