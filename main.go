package main

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"golang.org/x/term"

	"github.com/faiface/beep"
	"github.com/faiface/beep/speaker"
)

type Switcher struct {
	mu      sync.RWMutex
	buffers []*beep.Buffer
	cur     int // current buffer index
	pos     int // current sample index (frames)

	xfadeTotal int
	xfadeLeft  int

	fromCur int
	toCur   int
	fromPos int
	toPos   int

	tmpA [][2]float64
	tmpB [][2]float64
}

func (s *Switcher) startTransition(toCur int, toPos int) {
	// If we're already mid-fade, treat "current" as the fade target.
	if s.xfadeLeft > 0 {
		s.fromCur = s.toCur
		s.fromPos = s.toPos
	} else {
		s.fromCur = s.cur
		s.fromPos = s.pos
	}

	s.toCur = toCur
	s.toPos = toPos

	s.xfadeLeft = s.xfadeTotal

	// Make the "active" selection be the target; Stream() will mix while fading.
	s.cur = s.toCur
	s.pos = s.toPos
}

func (s *Switcher) Add(delta int) {
	s.mu.Lock()
	n := len(s.buffers)
	next := (s.cur + delta) % n
	if next < 0 {
		next += n
	}

	// crossfade from current -> next at the same sample index (synced files)
	s.startTransition(next, s.pos)

	s.mu.Unlock()
}

func (s *Switcher) Seek(deltaFrames int) {
	s.mu.Lock()

	target := s.pos + deltaFrames
	if target < 0 {
		target = 0
	}
	bufLen := s.buffers[s.cur].Len()
	if target > bufLen {
		target = bufLen
	}

	// crossfade within the same track: old position -> new position
	s.startTransition(s.cur, target)

	s.mu.Unlock()
}

func readFromBufferLoop(buf *beep.Buffer, pos *int, out [][2]float64) int {
	if len(out) == 0 {
		return 0
	}
	bufLen := buf.Len()
	if bufLen <= 0 {
		for i := range out {
			out[i][0], out[i][1] = 0, 0
		}
		return len(out)
	}

	written := 0
	for written < len(out) {
		if *pos >= bufLen {
			*pos = 0
		}
		remain := len(out) - written
		chunk := bufLen - *pos
		if chunk > remain {
			chunk = remain
		}
		st := buf.Streamer(*pos, *pos+chunk)
		n, _ := st.Stream(out[written : written+chunk])
		written += n
		*pos += n
		if n < chunk {
			for i := written; i < len(out); i++ {
				out[i][0], out[i][1] = 0, 0
			}
			return len(out)
		}
	}
	return written
}

func (s *Switcher) Stream(samples [][2]float64) (n int, ok bool) {
	s.mu.Lock()

	// Crossfade path (used for both track switches and seeks).
	if s.xfadeLeft > 0 && s.xfadeTotal > 0 {
		k := len(samples)
		if s.xfadeLeft < k {
			k = s.xfadeLeft
		}

		if cap(s.tmpA) < k {
			s.tmpA = make([][2]float64, k)
			s.tmpB = make([][2]float64, k)
		}
		a := s.tmpA[:k]
		b := s.tmpB[:k]

		fromBuf := s.buffers[s.fromCur]
		toBuf := s.buffers[s.toCur]

		fp := s.fromPos
		tp := s.toPos

		readFromBufferLoop(fromBuf, &fp, a)
		readFromBufferLoop(toBuf, &tp, b)

		start := s.xfadeTotal - s.xfadeLeft // frames already faded
		for i := 0; i < k; i++ {
			t := float64(start+i) / float64(s.xfadeTotal) // 0..1
			ga := math.Cos(t * math.Pi * 0.5)
			gb := math.Sin(t * math.Pi * 0.5)
			samples[i][0] = ga*a[i][0] + gb*b[i][0]
			samples[i][1] = ga*a[i][1] + gb*b[i][1]
		}

		s.fromPos = fp
		s.toPos = tp
		s.pos = tp
		s.cur = s.toCur

		s.xfadeLeft -= k

		// Fill the rest (if any) from the target stream (looping).
		if k < len(samples) {
			buf := s.buffers[s.cur]
			p := s.pos
			readFromBufferLoop(buf, &p, samples[k:])
			s.pos = p
		}

		s.mu.Unlock()
		return len(samples), true
	}

	// Normal path: loop forever.
	buf := s.buffers[s.cur]
	p := s.pos
	readFromBufferLoop(buf, &p, samples)
	s.pos = p

	s.mu.Unlock()
	return len(samples), true
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

	if ch != 1 && ch != 2 {
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
	if len(raw) == 0 || len(raw)%4 != 0 {
		return nil, beep.Format{}, fmt.Errorf("invalid ffmpeg output for %s", path)
	}

	totalF32 := len(raw) / 4
	frames := totalF32 / ch
	if frames <= 0 {
		return nil, beep.Format{}, fmt.Errorf("decoded 0 frames from %s", path)
	}

	format := beep.Format{
		SampleRate:  beep.SampleRate(sr),
		NumChannels: 2,
		Precision:   2,
	}

	if want != nil {
		if format.SampleRate != want.SampleRate || format.NumChannels != want.NumChannels {
			return nil, beep.Format{}, fmt.Errorf("format mismatch for %s", path)
		}
	}

	var idx int
	streamer := beep.StreamerFunc(func(out [][2]float64) (n int, ok bool) {
		remaining := frames - (idx / ch)
		if remaining <= 0 {
			return 0, false
		}
		max := len(out)
		if remaining < max {
			max = remaining
		}
		for i := 0; i < max; i++ {
			if ch == 1 {
				v := f32From(raw, idx)
				idx++
				out[i][0], out[i][1] = float64(v), float64(v)
			} else {
				l := f32From(raw, idx)
				r := f32From(raw, idx+1)
				idx += 2
				out[i][0], out[i][1] = float64(l), float64(r)
			}
		}
		return max, true
	})

	buf := beep.NewBuffer(format)
	buf.Append(streamer)
	if buf.Len() == 0 {
		return nil, beep.Format{}, fmt.Errorf("decoded 0 frames from %s", path)
	}
	return buf, format, nil
}

func f32From(b []byte, i int) float32 {
	off := i * 4
	return math.Float32frombits(binary.LittleEndian.Uint32(b[off:]))
}

type Config struct {
	SeekSeconds float64             `json:"seek_seconds"`
	Bindings    map[string][]string `json:"bindings"`
}

func defaultConfig() Config {
	return Config{
		SeekSeconds: 5,
		Bindings: map[string][]string{
			"next":            {"right"},
			"prev":            {"left"},
			"seek_forward":    {"up"},
			"seek_backward":   {"down"},
			"toggle_filename": {"f"},
			"quit":            {"q", "Q"},
		},
	}
}

// loads config if path exists; otherwise returns defaults.
// if path is empty, tries ./ab_test.json (optional).
func loadConfig(path string) (Config, error) {
	cfg := defaultConfig()

	// Optional auto-default: if no --config provided, try ./ab_test.json
	if path == "" {
		path = "ab_test.json"
	}

	b, err := os.ReadFile(path)
	if err != nil {
		// if file doesn't exist, just use defaults
		if os.IsNotExist(err) {
			return cfg, nil
		}
		return cfg, err
	}

	var userCfg Config
	if err := json.Unmarshal(b, &userCfg); err != nil {
		return cfg, fmt.Errorf("parse %s: %w", path, err)
	}

	// merge
	if userCfg.SeekSeconds > 0 {
		cfg.SeekSeconds = userCfg.SeekSeconds
	}
	if userCfg.Bindings != nil {
		for action, keys := range userCfg.Bindings {
			if len(keys) > 0 {
				cfg.Bindings[action] = keys
			}
		}
	}

	return cfg, nil
}

func buildKeymap(cfg Config) map[string]string {
	// key -> action
	m := make(map[string]string)
	for action, keys := range cfg.Bindings {
		for _, k := range keys {
			if k == "" {
				continue
			}
			m[k] = action
		}
	}
	return m
}

func parseKey(buf []byte, n int) string {
	if n <= 0 {
		return ""
	}
	// Arrow keys: ESC [ A/B/C/D
	if n >= 3 && buf[0] == 0x1b && buf[1] == '[' {
		switch buf[2] {
		case 'A':
			return "up"
		case 'B':
			return "down"
		case 'C':
			return "right"
		case 'D':
			return "left"
		}
	}
	// Single-byte keys
	if n == 1 {
		return string(buf[0])
	}
	return ""
}

func main() {
	var startIndex int
	var showFilename bool
	var verbose bool
	var configPath string
	var noShuffle bool
	flag.IntVar(&startIndex, "i", 0, "start file index")
	flag.BoolVar(&showFilename, "show-filename", false, "show filenames while switching (NOT blind)")
	flag.BoolVar(&verbose, "verbose", false, "print ffprobe/ffmpeg diagnostics to stderr")
	flag.StringVar(&configPath, "config", "", "path to config file (json). default: ./ab_test.json if present")
	flag.BoolVar(&noShuffle, "no-shuffle", false, "do not randomize the file order (default: shuffle)")
	flag.Parse()
	paths := flag.Args()

	if len(paths) < 2 {
		fmt.Fprintf(os.Stderr, "Usage: %s [-i startIndex] [--show-filename] [--verbose] [--config path.json] [--no-shuffle] file1 file2 [file3...]\n", os.Args[0])
		os.Exit(2)
	}

	seed := time.Now().UnixNano()
	r := rand.New(rand.NewSource(seed))

	// Shuffle by default (unless --no-shuffle).
	if !noShuffle {
		r.Shuffle(len(paths), func(i, j int) { paths[i], paths[j] = paths[j], paths[i] })
	}

	// Resolve the effective config path we will monitor.
	effectiveConfigPath := configPath
	if effectiveConfigPath == "" {
		effectiveConfigPath = "ab_test.json"
	}

	cfg, err := loadConfig(effectiveConfigPath)
	if err != nil {
		log.Fatal(err)
	}

	// Hot-reload state (atomics so the audio/input threads can read without locks).
	var keymapAtomic atomic.Value // holds map[string]string
	keymapAtomic.Store(buildKeymap(cfg))

	var jumpFramesAtomic int64 // frames to jump for seek
	var seekSecondsAtomic uint64
	atomic.StoreUint64(&seekSecondsAtomic, math.Float64bits(cfg.SeekSeconds))

	// runtime filename visibility toggle (initialised from flag)
	var showFilenameAtomic atomic.Bool
	showFilenameAtomic.Store(showFilename)

	// First decode establishes format (and starts speaker).
	firstBuf, format, err := decodeToBufferFFmpegRawFloat32(paths[0], nil, verbose)
	if err != nil {
		log.Fatal(err)
	}

	// Compute jump frames from seek seconds.
	jf := int64(float64(format.SampleRate) * cfg.SeekSeconds)
	if jf < 1 {
		jf = 1
	}
	atomic.StoreInt64(&jumpFramesAtomic, jf)

	bufferSize := format.SampleRate.N(50e6)
	if bufferSize < 1024 {
		bufferSize = 1024
	}
	speaker.Init(format.SampleRate, bufferSize)

	// Load remaining files.
	buffers := make([]*beep.Buffer, 0, len(paths))
	buffers = append(buffers, firstBuf)

	for _, p := range paths[1:] {
		buf, _, err := decodeToBufferFFmpegRawFloat32(p, &format, verbose)
		if err != nil {
			log.Fatal(err)
		}
		buffers = append(buffers, buf)
	}

	switcher := &Switcher{
		buffers: buffers,
		cur:     0,
		pos:     0,
		// short equal-power crossfade to avoid clicks on switches/seeks
		xfadeTotal: int(format.SampleRate) * 5 / 1000, // 5ms
	}
	if switcher.xfadeTotal < 1 {
		switcher.xfadeTotal = 1
	}

	if startIndex >= 0 && startIndex < len(switcher.buffers) {
		switcher.cur = startIndex
	}

	// Hot-reload goroutine: watch mtime and apply new bindings/seek_seconds.
	// (No extra deps like fsnotify; simple polling is reliable across platforms.)
	go func() {
		t := time.NewTicker(400 * time.Millisecond)
		defer t.Stop()

		var lastMod time.Time
		for range t.C {
			st, err := os.Stat(effectiveConfigPath)
			if err != nil {
				// If it doesn't exist, keep waiting (user may create it later).
				if os.IsNotExist(err) {
					lastMod = time.Time{}
					continue
				}
				continue
			}

			mod := st.ModTime()
			if !mod.After(lastMod) {
				continue
			}

			newCfg, err := loadConfig(effectiveConfigPath)
			if err != nil {
				if verbose {
					fmt.Fprintf(os.Stderr, "config reload failed: %v\n", err)
				}
				continue
			}

			keymapAtomic.Store(buildKeymap(newCfg))
			atomic.StoreUint64(&seekSecondsAtomic, math.Float64bits(newCfg.SeekSeconds))

			newJf := int64(float64(format.SampleRate) * newCfg.SeekSeconds)
			if newJf < 1 {
				newJf = 1
			}
			atomic.StoreInt64(&jumpFramesAtomic, newJf)

			lastMod = mod

			if verbose {
				fmt.Fprintf(os.Stderr, "reloaded config from %s (seek_seconds=%.3g)\n", effectiveConfigPath, newCfg.SeekSeconds)
			}
		}
	}()

	// Print initial UI.
	seekSeconds := math.Float64frombits(atomic.LoadUint64(&seekSecondsAtomic))
	fmt.Println("Controls: (configurable) switch/seek/quit via config")
	fmt.Printf("Loaded %d files. Format: %d Hz, %d ch\n", len(paths), format.SampleRate, format.NumChannels)
	fmt.Printf("Seek step: ±%.3g s\n", seekSeconds)
	if !noShuffle {
		fmt.Printf("Order: shuffled (seed=%d)\n", seed)
	} else {
		fmt.Println("Order: as provided (--no-shuffle)")
	}
	if showFilenameAtomic.Load() {
		fmt.Printf("Start: [%d] %s\n", switcher.cur, paths[switcher.cur])
	}

	speaker.Play(switcher)

	oldState, _ := term.MakeRaw(int(os.Stdin.Fd()))
	defer term.Restore(int(os.Stdin.Fd()), oldState)

	keybuf := make([]byte, 16)
	for {
		n, _ := os.Stdin.Read(keybuf)
		key := parseKey(keybuf, n)
		if key == "" {
			continue
		}

		kmAny := keymapAtomic.Load()
		if kmAny == nil {
			continue
		}
		keymap := kmAny.(map[string]string)

		action := keymap[key]
		if action == "" {
			continue
		}

		if action == "quit" {
			fmt.Println("\nBye.")
			return
		}

		jumpFrames := int(atomic.LoadInt64(&jumpFramesAtomic))

		speaker.Lock()
		switch action {
		case "prev":
			switcher.Add(-1)
		case "next":
			switcher.Add(+1)
		case "seek_forward":
			switcher.Seek(+jumpFrames)
		case "seek_backward":
			switcher.Seek(-jumpFrames)
		case "toggle_filename":
			showFilenameAtomic.Store(!showFilenameAtomic.Load())
		}

		// Refresh filename line if enabled
		if showFilenameAtomic.Load() && (action == "prev" || action == "next" || action == "toggle_filename") {
			fmt.Printf("\rNow: [%d] %s            ", switcher.cur, paths[switcher.cur])
		}
		if !showFilenameAtomic.Load() && action == "toggle_filename" {
			fmt.Print("\r                              ")
		}
		speaker.Unlock()
	}
}
