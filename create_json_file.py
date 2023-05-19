
import os
import glob
import json
import argparse


def create_file(tile_path, tile_format, write_path, json_name):

    tiles = glob.glob(os.path.join(tile_path, '*'+tile_format))
    tiles_list = []
    for tile in tiles:
        tiles_list.append({'image_name': tile})

    out_path = os.path.join(write_path, json_name+'.json')
    out_file = open(out_path, 'w')
    json.dump(tiles_list, out_file)
    out_file.close()


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='feed the tiles path to create the json file.')
    parser.add_argument('--tilePath', type=str, default='Z:/Active/mahsa/quality_groups/revised-groups/Q3', help='define the tile path')
    parser.add_argument('--tileFormat', default='.tif', type=str, help='define the format of the tile image')
    parser.add_argument('--writePath', type=str, default='./')
    parser.add_argument('--jsonName', type=str, default='EM_tiles_list', help='define the name of the json file')
    args = parser.parse_args()

    create_file(tile_path=args.tilePath, tile_format=args.tileFormat, write_path=args.writePath, json_name=args.jsonName)



