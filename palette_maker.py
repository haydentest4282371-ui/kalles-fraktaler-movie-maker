import pygame
import colorsys
import json

pygame.init()

W, H = 800, 400
screen = pygame.display.set_mode((W, H))
clock = pygame.time.Clock()

colors = []

repeat = 3
contour = 4
use_contour = True


def hsv_to_rgb(h, s=1, v=1):
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return [int(r * 255), int(g * 255), int(b * 255)]


def build_palette(colors, repeat=3, contour_repeat=4, use_contour=True):
    palette = []

    for c in colors:

        # one unit: C C C 0
        unit = []

        for _ in range(repeat):
            unit.append(c)

        if use_contour:
            unit.append([0, 0, 0])

        # repeat that unit 4 times (or contour_repeat times)
        for _ in range(contour_repeat):
            palette.extend(unit)

    return palette


def export_palette(palette, filename="palette.json"):
    with open(filename, "w") as f:
        json.dump(palette, f)


running = True
while running:
    screen.fill((20, 20, 20))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        # mouse controls
        if event.type == pygame.MOUSEBUTTONDOWN:
            x, y = pygame.mouse.get_pos()

            # left click = add color from hue bar
            if event.button == 1:
                hue = x / W
                colors.append(hsv_to_rgb(hue))

            # right click = remove last color
            if event.button == 3 and colors:
                colors.pop()

        # keyboard controls
        if event.type == pygame.KEYDOWN:

            # toggle contours
            if event.key == pygame.K_c:
                use_contour = not use_contour

            # adjust repeat
            if event.key == pygame.K_UP:
                repeat += 1

            if event.key == pygame.K_DOWN:
                repeat = max(1, repeat - 1)

            # adjust contour thickness
            if event.key == pygame.K_RIGHT:
                contour += 1

            if event.key == pygame.K_LEFT:
                contour = max(0, contour - 1)

            # export palette
            if event.key == pygame.K_s:
                palette = build_palette(colors, repeat, contour, use_contour)
                export_palette(palette)
                print("Saved palette.json")

    # --- draw hue bar ---
    for i in range(W):
        col = hsv_to_rgb(i / W)
        pygame.draw.line(screen, col, (i, 0), (i, 40))

    # --- preview palette ---
    palette = build_palette(colors, repeat, contour, use_contour)

    xoff = 0
    for c in palette[:100]:
        pygame.draw.rect(screen, c, (xoff, 80, 8, 40))
        xoff += 8

    # --- UI text ---
    font = pygame.font.SysFont(None, 24)
    txt = font.render(
        f"colors={len(colors)} repeat={repeat} contour={contour}  [C]=toggle contours  [S]=save",
        True,
        (255, 255, 255)
    )
    screen.blit(txt, (10, 150))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()