# Primary MazeGenerator application: one mesh (engine fallback quad), one
# custom fragment shader, no external assets. This is the starting point for
# the maze generator; add gameplay to App.cpp / App.cppm.
add_app(MazeGeneratorApp
    APP_TITLE        "MazeGenerator"
    APP_ID           "MazeGenerator"
    MODULE_PREFIX    MazeGenerator.App
    SHADER_NAMESPACE MazeGenerator
    SOURCES
        main.cpp
        App.cpp
    MODULES
        App.cppm
        TiledMaze.cppm
    SHADERS
        CheckerFrag checker.slang fragment main
)
