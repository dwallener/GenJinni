# top level
import os
from nicegui import ui, events

# Globals
upload_dir = 'uploads'
tileset_dir = f'{upload_dir}/tileset'
legality_dir = f'{upload_dir}/legality'

# Ensure directories exist
os.makedirs(tileset_dir, exist_ok=True)
os.makedirs(legality_dir, exist_ok=True)

# Helper function for upload handling
def handle_upload(e: events.UploadEventArguments, section, idx):
    b = e.content.read()
    outdir = tileset_dir if section == 'tileset' else legality_dir

    filename = f'{section}-{idx:02}.png'
    with open(os.path.join(outdir, filename), "wb") as file:
        file.write(b)

    ui.notify(f'File saved to {os.path.join(outdir, filename)}')

# UI setup
def create_home_page():
    with ui.row().style('justify-content: center'):
        ui.image('artwork/genjinn-large.png').style('height: 100px; width: auto;')

def create_content_generation_tab():
    ui.label('Upload Tileset (up to 16 tiles)').style('font-weight: bold')
    for i in range(16):
        ui.upload(on_upload=lambda e, idx=i: handle_upload(e, 'tileset', idx),
                  on_rejected=lambda e: ui.notify(f'Upload failed: {e.name}'),
                  label=f'Tile {i+1}').classes('max-w-full').props('accept=.png')

    ui.label('Upload Legality Samples (up to 8 samples)').style('font-weight: bold')
    for i in range(8):
        ui.upload(on_upload=lambda e, idx=i: handle_upload(e, 'legality', idx),
                  on_rejected=lambda e: ui.notify(f'Upload failed: {e.name}'),
                  label=f'Legality Sample {i+1}').classes('max-w-full').props('accept=.png')

    ui.button('Submit', on_click=lambda: ui.notify('Files submitted for processing')).style('margin-top: 20px')

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
        create_content_generation_tab()
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

