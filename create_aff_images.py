from util import *
from PIL import Image, ImageDraw, ImageFont

path = ''
SEED = 0
np.random.seed(SEED)
random.seed(SEED)

font_size = 16
for from_game in ['smb','ki','mm','met']:
    from_folder = path + 'data/' + game_folders_nor[from_game]
    from_levels_original, text = parse_folder(from_folder,from_game)
    from_levels_original_translated = []
    to_levels_generated_translated = []
    for i, from_level in enumerate(from_levels_original):
        from_translated = translate_level(from_level,from_game)
        from_level = [''.join(l) for l in from_level]
        from_img = get_image_from_segment(from_level)
        from_translated_text = '\n'.join(from_translated)
        font = ImageFont.truetype('courbd.ttf', font_size)
        aff_img = Image.new('RGB',(16*font_size, 15*font_size), color='white')
        aff_draw = ImageDraw.Draw(aff_img)
        for row, seq in enumerate(from_translated):
            for col, tile in enumerate(seq):
                aff_draw.text((col*font_size,row*font_size),tile,(0,0,0),font=font)
                
        from_img.save(f'output_seg/{from_game}-{i}.png')
        aff_img.save(f'output_seg/{from_game}-{i}-aff.png')
