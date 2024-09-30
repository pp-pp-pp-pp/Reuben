# Initialize Pygame
pygame.init()
infoObject = pygame.display.Info()
window_width, window_height = infoObject.current_w, infoObject.current_h

# Create a borderless window instead of exclusive fullscreen
screen = pygame.display.set_mode((window_width, window_height), pygame.NOFRAME)
pygame.display.set_caption("2D Pressure Field Simulation with Live Sound")
clock = pygame.time.Clock()

# Adjustable Parameters for Scale and Position
visualizer_scale = 8  # Controls the size of each grid cell (pixels)

# Position the visualizer as needed
# For a borderless window, you might not need to adjust positions relative to the screen
visualizer_position = (0, -350)  # Example for 1600x900 resolution
