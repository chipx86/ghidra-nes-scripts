# NES Disassembly Scripts for Ghidra

This is a set of scripts I've developed initially to aid in my
[Faxanadu for NES disassembly work](https://github.com/chipx86/faxanadu).

There are two scripts in this repository:

* `export_nes.py`
* `populate_ppu_tile_16.py`


## `export_nes.py`

**Menu Item:** Tools -> Export NES

This takes the disassembled banks for an NES ROM and turns it into:

1. An assembly source file compatible with
   [asm6f](https://github.com/freem/asm6f).

2. Browsable, annotated, pretty-printed HTML source that can be used to
   view the NES source with all comments and references included.


## `populate_ppu_tile_16.py`

**Menu Item:** Tools -> Visualize Sprites

This takes a 16-byte selection representing a 16-byte PPU tile and turns it
into an ASCII visualization attached to a comment. This helps with documenting
the tiles found within the ROM.


## Activation

Copy these scripts to your `$HOME/ghidra_scripts` (macOS/Linux) and activate the plugins. They'll then be available in the **Tools** menu.
