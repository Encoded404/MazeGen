# Primary MazeGenerator application: a fullscreen maze pass reading the packed
# TiledMaze bits directly, with no wall mesh, plus an incremental-solver debug
# overlay and an ImGui control panel.
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
        DebugOverlayPass.cpp
    MODULES
        App.cppm
        TiledMaze.cppm
        PackedCellGrid.cppm
        MazeDensity.cppm
        MazeRenderPass.cppm
        Generation.cppm
        generators/Random.cppm
        generators/CustomGeneratorV1.cppm
        Generators.cppm
        Solver.cppm
        solvers/AStar.cppm
        Solvers.cppm
        DebugGrid.cppm
        DebugOverlayPass.cppm
    SHADERS
        MazeFullscreenVert maze_fullscreen_vert.slang vertex   main
        MazeFrag           maze.slang                 fragment main
        DebugOverlayFrag   debug_overlay.slang        fragment main
)
