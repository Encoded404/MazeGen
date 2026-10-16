# sdl3 overlay: disable the dynamic-API layer (SDL_DYNAMIC_API -> 0)

SDL's dynapi jump table (one .data array of ~1270 function pointers in
src/dynapi/SDL_dynapi.c) references every public SDL function across all
subsystems. With a static SDL that table makes ALL of SDL unreachable to
--gc-sections (only ~34 KiB of ~2.5 MiB was prunable). Removing dynapi lets
gc-sections work as intended: measured -1,158,928 B (-11.4%) on the Release
`minimal` executable (GPU API, GL/Vulkan renderers and most of the 2D blit
suite then get compiled out at link time).

Dynapi only matters when SDL is loaded via dlopen / ABI-shimming; it is dead
weight for this project's static linking. SDL guards against disabling it from
the command line (#error), hence the header patch.
