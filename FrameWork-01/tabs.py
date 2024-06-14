from nicegui import ui

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
            with ui.tabs() as level_design_tabs:
                ui.tab('Tileset')
                ui.tab('Legality Samples')
                ui.tab('Generation Settings')

            with ui.tab_panels(level_design_tabs):
                with ui.tab_panel('Tileset'):
                    ui.label('Upload the tileset (up to 16 tiles)').style('font-weight: bold; color: blue')
                    with ui.grid(columns=4):
                        for i in range(16):
                            ui.upload(on_upload=lambda e, idx=i: handle_upload(e, 'tileset', idx),
                                      on_rejected=lambda e: ui.notify(f'Upload failed: {e.name}'),
                                      label=f'Tile {i+1}').classes('max-w-full').props('accept=.png')
                
                with ui.tab_panel('Legality Samples'):
                    ui.label('Upload Samples showing legal connectivity (up to 8 samples)').style('font-weight: bold; color: blue')
                    with ui.grid(columns=2):
                        for i in range(8):
                            ui.upload(on_upload=lambda e, idx=i: handle_upload(e, 'legality', idx),
                                      on_rejected=lambda e: ui.notify(f'Upload failed: {e.name}'),
                                      label=f'Legality Sample {i+1}').classes('max-w-full').props('accept=.png')
                
                with ui.tab_panel('Generation Settings'):
                    save_path = ui.input(label='Save inferred rules to:', placeholder='Enter path').style('margin-top: 20px')
                    ui.button('Infer placement rules', color='green', on_click=lambda: ui.notify(f'Saving to {save_path.value}')).style('font-size: 24px; margin-top: 20px')
                    
                    num_levels = ui.number(label='Number of levels to generate:').style('margin-top: 20px')
                    levels_save_path = ui.input(label='Save generated levels to:', placeholder='Enter path').style('margin-top: 20px')
                    ui.button('Generate Levels', color='green', on_click=lambda: ui.notify(f'Generating {num_levels.value} levels to {levels_save_path.value}')).style('font-size: 24px; margin-top: 20px')

        with ui.tab_panel('Enemy Design'):
            for character_idx in range(1, 4):
                ui.label(f'Define Character {character_idx}').style('font-weight: bold; color: blue')
                ui.input(label='Character name').style('margin-top: 20px')
                
                ui.label('Character Images').style('margin-top: 20px; font-weight: bold; color: blue')
                with ui.grid(columns=4):
                    for img_idx in range(4):
                        ui.upload(on_upload=lambda e, idx=img_idx: handle_upload(e, f'character{character_idx}', idx),
                                  on_rejected=lambda e: ui.notify(f'Upload failed: {e.name}'),
                                  label=f'Image {img_idx + 1}').classes('max-w-full').props('accept=.png')
                
                ui.label('Animation Keyframes').style('margin-top: 20px; font-weight: bold; color: blue')
                with ui.grid(columns=4):
                    for keyframe_idx in range(4):
                        ui.upload(on_upload=lambda e, idx=keyframe_idx: handle_upload(e, f'keyframe{character_idx}', idx),
                                  on_rejected=lambda e: ui.notify(f'Upload failed: {e.name}'),
                                  label=f'Keyframe {keyframe_idx + 1}').classes('max-w-full').props('accept=.png')

        with ui.tab_panel('Item and Upgrade Generation'):
            ui.label('Item and Upgrade Generation features go here.')

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