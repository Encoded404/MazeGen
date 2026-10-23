# Primary MazeGenerator application: a fullscreen maze pass reading the packed
# TiledMaze bits directly, with no wall mesh.
add_app(MazeGeneratorApp
    OUTPUT_SUBDIR    app
    FLAT_OUTPUT
    APP_TITLE        "MazeGenerator"
    APP_ID           "MazeGenerator"
    MODULE_PREFIX    MazeGenerator.App
    SHADER_NAMESPACE MazeGenerator
    SOURCES
        main.cpp
        App.cpp
        MazeRenderPass.cpp
    MODULES
        App.cppm
        TiledMaze.cppm
        MazeRenderPass.cppm
        MazeDensity.cppm
    SHADERS
        MazeFullscreenVert maze_fullscreen_vert.slang vertex   main
        MazeFrag           maze.slang                 fragment main
)
