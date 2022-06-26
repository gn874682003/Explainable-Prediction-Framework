import seaborn as sns
import matplotlib.pyplot as plt

# RGB color format conversion to Hex
def RGB_to_Hex(rgb):
    RGB = rgb.split(',')  # Separate RGB formats
    color = '#'
    for i in RGB:
        num = int(i)
        color += str(hex(num))[-2:].replace('x', '0').upper()
    # print(color)
    return color

# RGB color format conversion to Hex
def RGB_list_to_Hex(RGB):
    # RGB = rgb.split(',') Separate RGB formats
    color = '#'
    for i in RGB:
        num = int(i)
        color += str(hex(num))[-2:].replace('x', '0').upper()
    # print(color)
    return color

# Hex color format conversion to RGB
def Hex_to_RGB(hex):
    r = int(hex[1:3], 16)
    g = int(hex[3:5], 16)
    b = int(hex[5:7], 16)
    rgb = str(r) + ',' + str(g) + ',' + str(b)
    # print(rgb)
    return rgb, [r, g, b]

# Generate  gradient
def gradient_color(color_list, color_sum=20):
    color_center_count = len(color_list)
    # if color_center_count == 2:
    #     color_center_count = 1
    color_sub_count = int(color_sum / (color_center_count - 1))
    color_index_start = 0
    color_map = []
    for color_index_end in range(1, color_center_count):
        color_rgb_start = Hex_to_RGB(color_list[color_index_start])[1]
        color_rgb_end = Hex_to_RGB(color_list[color_index_end])[1]
        r_step = (color_rgb_end[0] - color_rgb_start[0]) / color_sub_count
        g_step = (color_rgb_end[1] - color_rgb_start[1]) / color_sub_count
        b_step = (color_rgb_end[2] - color_rgb_start[2]) / color_sub_count
        # Generate middle gradient
        now_color = color_rgb_start
        color_map.append(RGB_list_to_Hex(now_color))
        for color_index in range(1, color_sub_count):
            now_color = [now_color[0] + r_step, now_color[1] + g_step, now_color[2] + b_step]
            color_map.append(RGB_list_to_Hex(now_color))
        color_index_start = color_index_end
    return color_map

if __name__ == '__main__':
    # input_colors = ["#40FAFF", "#00EBEB", "#00EB00", "#FFC800", "#FC9600", "#FA0000", "#C800FA", "#FF64FF"]
    input_colors = ["#4682B4", "#FFFAFA"]
    colors = gradient_color(input_colors)
    sns.palplot(colors)
    plt.show()