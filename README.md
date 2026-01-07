# ab_test

CLI A/B switcher for **blind, sample-synced audio comparisons**.

- Instant switching at the same sample position (assumes files are sample-synced)
- Click-free switches/seeks via short (~5ms) equal-power crossfade
- Loops forever (until quit)
- Uses `ffmpeg`/`ffprobe` to decode (so most formats work)
- Configurable keybinds + seek step (hot-reloaded)
- Shuffles input order by default (use `--no-shuffle` to disable)
- **Does NOT show filenames by default** (blind). Can be enabled/toggled.
- Minimal startup output by default; use `-info` for full banner

## Requirements

- Go
- `ffmpeg` + `ffprobe` in `PATH`

## Build

```sh
go build
````

## Usage

```sh
./ab_test [--config ab_test.json] [--no-shuffle] [--show-filename] [-info] [--verbose] [-i N] file1 file2 [file3...]
```

By default, startup output is limited to:

```
Loaded N files. Output: SR Hz, CH ch
```

Use `-info` to also show controls, seek step, resampling mode, and shuffle order.

## Config

If `./ab_test.json` exists, it’s loaded automatically (or use `--config path.json`).
Edits are hot-reloaded while running.

Example:

```json
{
  "seek_seconds": 5,
  "bindings": {
    "next": ["right"],
    "prev": ["left"],
    "seek_forward": ["up"],
    "seek_backward": ["down"],
    "toggle_filename": ["f"],
    "quit": ["q", "Q"]
  }
}
```

Key names:

* arrows: `up`, `down`, `left`, `right`
* letters: `"a"`, `"F"`, etc.

Actions:

* `next`, `prev`
* `seek_forward`, `seek_backward`
* `toggle_filename`
* `quit`
