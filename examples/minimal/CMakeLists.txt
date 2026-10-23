# Minimal example: one mesh (engine fallback quad), one custom fragment
# shader, no external assets. The primary app lives in app/; copy this directory
# to add another example and rename the target, MODULE_PREFIX and
# SHADER_NAMESPACE.
add_app(minimal
    APP_TITLE        "MazeGenerator Minimal"
    MODULE_PREFIX    Examples.Minimal
    SHADER_NAMESPACE Minimal
    SOURCES
        main.cpp
        Game.cpp
    MODULES
        Game.cppm
    SHADERS
        CheckerFrag  checker.slang  fragment  main
)
