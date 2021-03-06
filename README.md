# GPU Game of Life
A graphical depiction of Conway’s Game of Life written in C++.

## Features
* [ ] Pause and un-pause the simulation
* [ ] Click on cells to toggle their state
* [ ] Zoom in and out
* [ ] Clear the existing grid
* [ ] Randomize the existing grid
* [ ] Display current FPS
* [ ] Pre-built test cases
* [ ] Stretch Goal: Control the speed of the simulation
* [ ] Stretch Goal: Deploy “templates” of commonly-known entities in the Game of Life, e.g. gliders

## Implementation Details
All update operations will occur on the GPU. This will allow for huge grid sizes to be updated quickly. Ideally, along with tile-based kernel configuration, we will divide the grid into “chunks” and only update portions that have any alive cells for maximum performance.
