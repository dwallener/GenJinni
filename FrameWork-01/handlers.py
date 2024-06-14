import os
from nicegui import events, ui

# Helper function for upload handling
def handle_upload(e: events.UploadEventArguments, section, idx):
    upload_dir = 'uploads'
    tileset_dir = f'{upload_dir}/tileset'
    legality_dir = f'{upload_dir}/legality'
    
    b = e.content.read()
    outdir = tileset_dir if section == 'tileset' else legality_dir

    filename = f'{section}-{idx:02}.png'
    with open(os.path.join(outdir, filename), "wb") as file:
        file.write(b)

    ui.notify(f'File saved to {os.path.join(outdir, filename)}')

