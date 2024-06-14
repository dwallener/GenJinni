from nicegui import ui
from level_design_tab import create_level_design_tab
from enemy_design_tab import create_enemy_design_tab
from item_upgrade_tab import create_item_upgrade_tab

def create_home_page():
    with ui.row().style('justify-content: center'):
        ui.image('artwork/genjinn-large.png').style('height: 100px; width: auto;')

def create_content_generation_tab(handle_upload):
    ui.label('Content Generation').style('font-weight: bold; font-size: 20px; color: blue')
    
    with ui.tabs() as sub_tabs:
        ui.tab('Level Design')
        ui.tab('Enemy Design')
        ui.tab('Item and Upgrade Generation')

    with ui.tab_panels(sub_tabs):
        with ui.tab_panel('Level Design'):
            create_level_design_tab(handle_upload)
        
        with ui.tab_panel('Enemy Design'):
            create_enemy_design_tab(handle_upload)

        with ui.tab_panel('Item and Upgrade Generation'):
            create_item_upgrade_tab()

def create_game_mechanics_tab():
    ui.label('Game Mechanics')

def create_nlp_game_logic_tab():
    ui.label('NLP for Game Logic')

def create_art_animation_tab():
    ui.label('Art & Animation')

def create_voice_sound_effects_tab():
    ui.label('Voice and Sound Effects')

def create_component_integration_tab():
    ui.label('Component Integration')

def create_generate_tab():
    with ui.row().style('justify-content: center; align-items: center; height: 100vh;'):
        ui.button('Generate the Game', color='red').style('font-size: 24px; padding: 20px 40px; border-radius: 10px;')