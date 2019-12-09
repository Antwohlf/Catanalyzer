import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from PIL import ImageDraw, ImageFont
import os

reds = Image.open("../GameIcons/reds.png")
oranges = Image.open("../GameIcons/oranges.png")
blues = Image.open("../GameIcons/blues.png")
whites = Image.open("../GameIcons/whites.png")
dice_img = Image.open("../GameIcons/dice.png")

town_range = [(100,260),(10,130)]
user_range = [(130,260),(160,256)]
city_range = [(250,479),(40,236)]
road_range = [(20,90),(40,236)]

red_town = reds.crop((town_range[1][0], town_range[0][0], town_range[1][1], town_range[0][1]))
red_user = reds.crop((user_range[1][0], user_range[0][0], user_range[1][1], user_range[0][1]))
red_city = reds.crop((city_range[1][0], city_range[0][0], city_range[1][1], city_range[0][1])).transpose(Image.FLIP_LEFT_RIGHT)
red_road = reds.crop((road_range[1][0], road_range[0][0], road_range[1][1], road_range[0][1]))

orange_town = oranges.crop((town_range[1][0], town_range[0][0], town_range[1][1], town_range[0][1]))
orange_user = oranges.crop((user_range[1][0], user_range[0][0], user_range[1][1], user_range[0][1]))
orange_city = oranges.crop((city_range[1][0], city_range[0][0], city_range[1][1], city_range[0][1])).transpose(Image.FLIP_LEFT_RIGHT)
orange_road = oranges.crop((road_range[1][0], road_range[0][0], road_range[1][1], road_range[0][1]))

blue_town = blues.crop((town_range[1][0], town_range[0][0], town_range[1][1], town_range[0][1]))
blue_user = blues.crop((user_range[1][0], user_range[0][0], user_range[1][1], user_range[0][1]))
blue_city = blues.crop((city_range[1][0], city_range[0][0], city_range[1][1], city_range[0][1])).transpose(Image.FLIP_LEFT_RIGHT)
blue_road = blues.crop((road_range[1][0], road_range[0][0], road_range[1][1], road_range[0][1]))

white_town = whites.crop((town_range[1][0], town_range[0][0], town_range[1][1], town_range[0][1]))
white_user = whites.crop((user_range[1][0], user_range[0][0], user_range[1][1], user_range[0][1]))
white_city = whites.crop((city_range[1][0], city_range[0][0], city_range[1][1], city_range[0][1])).transpose(Image.FLIP_LEFT_RIGHT)
white_road = whites.crop((road_range[1][0], road_range[0][0], road_range[1][1], road_range[0][1]))

scale_factor = 0.2
road_spacing = (40, 100)
town_spacing = (20, 30)
city_spacing = (40, 60)

white_player_start = (20, 80)
red_player_start = (20, 230)
orange_player_start = (20, 380)
blue_player_start = (20, 530)

def scale_down(img, percent):
    out = img.resize( [int(percent * s) for s in img.size] )
    return out

def sum_tuples(tuple1, tuple2):
    return tuple(map(lambda x, y: x + y, tuple1, tuple2))

def overlay_dice(dice, overlay):
    start = (10, 10)
    fonts_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'fonts')
    font = ImageFont.truetype(os.path.join(fonts_path, 'garamond-mt-bold_bigfontsite.com.ttf'), 50)

    if(dice > 1):
        overlay.alpha_composite(scale_down(dice_img, 0.4), start)
        draw = ImageDraw.Draw(overlay)
        draw.text((65, 5), str(dice), (220, 220, 220), font=font)
    return overlay


def overlay_stats(stats, color, overlay):
    town = red_town
    city = red_city
    road = red_road
    user = red_user
    start = red_player_start

    if(color == "blue"):
        town = blue_town
        city = blue_city
        road = blue_road
        user = blue_user
        start = blue_player_start

    if(color == "orange"):
        town = orange_town
        city = orange_city
        road = orange_road
        user = orange_user
        start = orange_player_start

    if(color == "white"):
        town = white_town
        city = white_city
        road = white_road
        user = white_user
        start = white_player_start

    overlay.alpha_composite(scale_down(user, scale_factor), start)

    road_cols = 4

    for i in range(stats["road"]):
        row = i // road_cols
        overlay.alpha_composite(scale_down(road, scale_factor), sum_tuples(start, (road_spacing[0] * (i % road_cols), road_spacing[1] + (10 * row))))

    for i in range(stats["town"]):
        overlay.alpha_composite(scale_down(town, scale_factor), sum_tuples(start, (town_spacing[0] * i, town_spacing[1])))

    for i in range(stats["city"]):
        overlay.alpha_composite(scale_down(city, scale_factor), sum_tuples(start, (city_spacing[0] * i, city_spacing[1])))
    return overlay

def makeOverlay(frame, features):    
    # overlay = np.zeros(frame.shape)
    overlayImage = Image.fromarray(frame, 'RGBA')
    overlayImage = overlay_stats(features["white"], "white", overlayImage)
    overlayImage = overlay_stats(features["red"], "red", overlayImage)
    overlayImage = overlay_stats(features["blue"], "blue", overlayImage)
    overlayImage = overlay_stats(features["orange"], "orange", overlayImage)
    overlayImage = overlay_dice(features["dice"], overlayImage)
    return overlayImage

def main():
    frame = np.zeros((720, 1080, 3))
    features = {"red":{"road":9, "town":3, "city":2}, "white":{"road":5, "town":2, "city":0}, "blue":{"road":12, "town":4, "city":2}, "orange":{"road":3, "town":4, "city":5}}
    makeOverlay(frame, features)

if __name__ == "__main__":
    main()