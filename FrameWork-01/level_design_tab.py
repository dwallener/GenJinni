from nicegui import ui

def create_level_design_tab(handle_upload):
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


