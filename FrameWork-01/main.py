import os
from nicegui import ui
from handlers import handle_upload
from tabs import create_home_page, create_content_generation_tab, create_game_mechanics_tab, create_nlp_game_logic_tab, create_art_animation_tab, create_voice_sound_effects_tab, create_component_integration_tab, create_generate_tab

# Ensure directories exist
upload_dir = 'uploads'
tileset_dir = f'{upload_dir}/tileset'
legality_dir = f'{upload_dir}/legality'
os.makedirs(tileset_dir, exist_ok=True)
os.makedirs(legality_dir, exist_ok=True)

with ui.tabs() as tabs:
    ui.tab('Home', icon='home')
    ui.tab('Content Generation', icon='build')
    ui.tab('Game Mechanics', icon='settings')
    ui.tab('NLP for Game Logic', icon='text_fields')
    ui.tab('Art & Animation', icon='palette')
    ui.tab('Voice and Sound Effects', icon='volume_up')
    ui.tab('Component Integration', icon='integration_instructions')
    ui.tab('Generate', icon='play_arrow')

with ui.tab_panels(tabs, value='Home'):
    with ui.tab_panel('Home'):
        create_home_page()
    with ui.tab_panel('Content Generation'):
        create_content_generation_tab(handle_upload)
    with ui.tab_panel('Game Mechanics'):
        create_game_mechanics_tab()
    with ui.tab_panel('NLP for Game Logic'):
        create_nlp_game_logic_tab()
    with ui.tab_panel('Art & Animation'):
        create_art_animation_tab()
    with ui.tab_panel('Voice and Sound Effects'):
        create_voice_sound_effects_tab()
    with ui.tab_panel('Component Integration'):
        create_component_integration_tab()
    with ui.tab_panel('Generate'):
        create_generate_tab()

ui.run()