add_app(basic_scene
    APP_TITLE       "MazeGenerator Basic Scene"
    MODULE_PREFIX   Examples.BasicScene
    SHADER_NAMESPACE BasicScene
    SOURCES
        main.cpp
        Game.cpp
    MODULES
        Game.cppm
        Components/SimpleControllerComponent.cppm
        Components/TransformControlComponent.cppm
    SHADERS
        NormalsFrag  normals.slang  fragment main
        SolidFrag    solid.slang    fragment main
    ASSET_DIRS
        models
        textures
)