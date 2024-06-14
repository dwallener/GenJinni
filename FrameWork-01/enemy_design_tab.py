from nicegui import ui

def create_enemy_design_tab(handle_upload):
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

