import pygame
import os
import math

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
 
    # Method used to display X and Y coordinates
    # of a point
    def displayPoint(self, p):
        print(f"({p.x}, {p.y})")
    
    def __repr__(self) -> str:
        return(f'Point({self.x}, {self.y})')

class Line_Segment:
    def __init__(self, start_point, end_point):
        self.A = start_point
        self.B = end_point

    def lineLineIntersection(self, other):
        # Line AB represented as a1x + b1y = c1
        a1 = self.B.y - self.A.y
        b1 = self.A.x - self.B.x
        c1 = a1*(self.A.x) + b1*(self.A.y)
    
        # Line CD represented as a2x + b2y = c2
        a2 = other.B.y - other.A.y
        b2 = other.A.x - other.B.x
        c2 = a2*(other.A.x) + b2*(other.A.y)
    
        determinant = a1*b2 - a2*b1
    
        if (determinant == 0):
            # The lines are parallel. This is simplified
            # by returning a pair of FLT_MAX
            return -1
        else:
            x = (b2*c1 - b1*c2)/determinant
            y = (a1*c2 - a2*c1)/determinant
            return Point(x, y)
        
    def onSegment(self, p, q, r):
        if ( (q.x <= max(p.x, r.x)) and (q.x >= min(p.x, r.x)) and 
            (q.y <= max(p.y, r.y)) and (q.y >= min(p.y, r.y))):
            return True
        return False
    
    def orientation(self, p, q, r):
        # to find the orientation of an ordered triplet (p,q,r)
        # function returns the following values:
        # 0 : Collinear points
        # 1 : Clockwise points
        # 2 : Counterclockwise
        
        # See https://www.geeksforgeeks.org/orientation-3-ordered-points/amp/ 
        # for details of below formula. 
        
        val = (float(q.y - p.y) * (r.x - q.x)) - (float(q.x - p.x) * (r.y - q.y))
        if (val > 0):
            
            # Clockwise orientation
            return 1
        elif (val < 0):
            
            # Counterclockwise orientation
            return 2
        else:
            
            # Collinear orientation
            return 0
    # The main function that returns true if 
    # the line segment 'p1q1' and 'p2q2' intersect.
    def doIntersect(self, other):
        
        # Find the 4 orientations required for 
        # the general and special cases
        p1 = self.A
        q1 = self.B
        p2 = other.A
        q2 = other.B
        o1 = self.orientation(p1, q1, p2)
        o2 = self.orientation(p1, q1, q2)
        o3 = self.orientation(p2, q2, p1)
        o4 = self.orientation(p2, q2, q1)
    
        # General case
        if ((o1 != o2) and (o3 != o4)):
            return True
    
        # Special Cases
    
        # p1 , q1 and p2 are collinear and p2 lies on segment p1q1
        if ((o1 == 0) and self.onSegment(p1, p2, q1)):
            return True
    
        # p1 , q1 and q2 are collinear and q2 lies on segment p1q1
        if ((o2 == 0) and self.onSegment(p1, q2, q1)):
            return True
    
        # p2 , q2 and p1 are collinear and p1 lies on segment p2q2
        if ((o3 == 0) and self.onSegment(p2, p1, q2)):
            return True
    
        # p2 , q2 and q1 are collinear and q1 lies on segment p2q2
        if ((o4 == 0) and self.onSegment(p2, q1, q2)):
            return True
    
        # If none of the cases
        return False 

class HalfLine:
    def __init__(self, origin, direction_point):
        self.origin = origin
        self.direction_point = direction_point
        self.slope, self.intercept = self.calculate_slope_and_intercept()

    def calculate_slope_and_intercept(self):
        slope = (self.direction_point.y - self.origin.y) / (self.direction_point.x - self.origin.x)
        intercept = self.origin.y - slope * self.origin.x
        return slope, intercept

def find_intersection(half_line, segment):
    def on_half_line(point):
        return (point.x - half_line.origin.x) * (half_line.direction_point.x - half_line.origin.x) >= 0 and \
               (point.y - half_line.origin.y) * (half_line.direction_point.y - half_line.origin.y) >= 0

    def on_line(point):
        return point.y == half_line.slope * point.x + half_line.intercept

    def between(x, a, b):
        return a <= x <= b or b <= x <= a

    segment_start = segment.A
    segment_end = segment.B

    if on_half_line(segment_start) and on_half_line(segment_end):
        if on_line(segment_start) and on_line(segment_end):
            if between(segment_start.x, segment_end.x, segment_start.x) and between(segment_start.y, segment_end.y, segment_start.y):
                return segment_start
        else:
            return segment_start

    x_intersection = (half_line.intercept - segment_start.y + half_line.slope * segment_start.x) / half_line.slope
    y_intersection = half_line.slope * x_intersection + half_line.intercept

    if between(x_intersection, segment_start.x, segment_end.x) and between(y_intersection, segment_start.y, segment_end.y):
        return Point(x_intersection, y_intersection)

    return None





def rotate_points(point1, point2, point3, point4, angle_degrees, rotation_offset):
    # Convert the angle from degrees to radians
    angle_rad = math.radians(angle_degrees)

    # Calculate the coordinates of the center of rotation
    center_x = rotation_offset[0]
    center_y = rotation_offset[1]

    # Calculate the sine and cosine of the angle
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    # Rotate each point around the center of rotation
    def rotate_point(x, y):
        x_rotated = center_x + cos_a * (x - center_x) - sin_a * (y - center_y)
        y_rotated = center_y + sin_a * (x - center_x) + cos_a * (y - center_y)
        return Point(x_rotated, y_rotated)

    rotated_point1 = rotate_point(point1.x, point1.y)
    rotated_point2 = rotate_point(point2.x, point2.y)
    rotated_point3 = rotate_point(point3.x, point3.y)
    rotated_point4 = rotate_point(point4.x, point4.y)

    return rotated_point1, rotated_point2, rotated_point3, rotated_point4

  
    

borders = [Line_Segment(Point(1356, 301), Point(898, 401)), Line_Segment(Point(1622, 355), Point(1356, 301)), Line_Segment(Point(1672, 395), Point(1622, 355)), Line_Segment(Point(1581, 521), Point(1672, 395)), Line_Segment(Point(1549, 824), Point(1581, 521)), Line_Segment(Point(1430, 896), Point(1549, 824)), Line_Segment(Point(1283, 862), Point(1430, 896)), Line_Segment(Point(1176, 629), Point(1283, 862)), Line_Segment(Point(1040, 490), Point(1176, 629)), Line_Segment(Point(564, 760), Point(1040, 490)), Line_Segment(Point(319, 730), Point(564, 760)), Line_Segment(Point(231, 671), Point(319, 730)), Line_Segment(Point(355, 614), Point(231, 671)), Line_Segment(Point(443, 472), Point(355, 614)), Line_Segment(Point(356, 299), Point(443, 472)), Line_Segment(Point(398, 236), Point(356, 299)), Line_Segment(Point(514, 186), Point(398, 236)), Line_Segment(Point(630, 189), Point(514, 186)), Line_Segment(Point(724, 306), Point(630, 189)), Line_Segment(Point(898, 395), Point(724, 306)), Line_Segment(Point(1006, 376), Point(898, 395)), Line_Segment(Point(1352, 125), Point(923, 216)), Line_Segment(Point(1696, 190), Point(1352, 125)), Line_Segment(Point(1916, 360), Point(1696, 190)), Line_Segment(Point(1749, 586), Point(1916, 360)), Line_Segment(Point(1743, 913), Point(1749, 586)), Line_Segment(Point(1462, 1074), Point(1743, 913)), Line_Segment(Point(1160, 1011), Point(1462, 1074)), Line_Segment(Point(1006, 701), Point(1160, 1011)), Line_Segment(Point(620, 944), Point(1006, 701)), Line_Segment(Point(257, 901), Point(620, 944)), Line_Segment(Point(82, 792), Point(257, 901)), Line_Segment(Point(3, 586), Point(82, 792)), Line_Segment(Point(240, 468), Point(3, 586)), Line_Segment(Point(153, 294), Point(240, 468)), Line_Segment(Point(282, 100), Point(153, 294)), Line_Segment(Point(484, 6), Point(282, 100)), Line_Segment(Point(714, 17), Point(484, 6)), Line_Segment(Point(836, 172), Point(714, 17)), Line_Segment(Point(919, 211), Point(836, 172)), Line_Segment(Point(1066, 181), Point(919, 211))]


# Initialize Pygame
pygame.init()


# Set up the display
screen_width = 1920 
screen_height = 1080 
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Car Game")

# Load images and resize car image
car_image = pygame.image.load("car.png")
car_width, car_height = 100, 50
car_image = pygame.transform.scale(car_image, (car_width, car_height))
car_rect = car_image.get_rect()
bg_image = pygame.image.load("bg1.png")
pozadi = pygame.image.load("Pozadi2.png")
priroda = pygame.image.load("Priroda.png")
bg_image = pygame.transform.scale(bg_image, (screen_width, screen_height))
pozadi = pygame.transform.scale(pozadi, (screen_width, screen_height))
priroda = pygame.transform.scale(priroda, (screen_width, screen_height))

# Car properties
car_x = (screen_width - car_width) // 2
car_y = screen_height - car_height - 20
car_speed = 0  # Current car speed
max_speed = 100  # Maximum speed of the car
acceleration = 0.3  # Acceleration rate

# Car angle (the direction it's facing) in degrees
car_angle = 0

# Adjustable turning speed
turn_speed_base = 5
turn_speed_adjust = 1  # Change this value to adjust the turning speed

# Friction
friction = 0.02

# Offset for the rotation point (half the width of the car)
rotation_offset = 0

# Game loop
clock = pygame.time.Clock()
is_running = True

def respawn_car():
    global car_x, car_y, car_speed, car_angle
    car_x = screen_width // 2
    car_y = 1080 // 8 + 100
    car_speed = 0
    car_angle = 0

screen.blit(pozadi, (0, 0))
screen.blit(priroda, (0, 0))
while is_running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            is_running = False
        
    

    
    # Handle user input
    keys = pygame.key.get_pressed()

    # Exit the game if "Escape" key is pressed
    if keys[pygame.K_ESCAPE]:
        is_running = False

    # Accelerate forward
    if keys[pygame.K_w]:
        car_speed = min(car_speed + acceleration, max_speed)

    # Slow down (decelerate) or move backward
    if keys[pygame.K_s]:
        car_speed = max(car_speed - acceleration, -max_speed)

    # Turn left
    if keys[pygame.K_a]:
        car_angle = (car_angle + (turn_speed_base * turn_speed_adjust)) % 360

    # Turn right
    if keys[pygame.K_d]:
        car_angle = (car_angle - (turn_speed_base * turn_speed_adjust)) % 360

    # Apply friction effect to gradually slow down the car
    car_speed *= (1 - friction)

    # Calculate acceleration vector based on car angle
    car_accel_x = car_speed * math.cos(math.radians(car_angle))
    car_accel_y = -car_speed * math.sin(math.radians(car_angle))

    # Update car position and speed
    car_x += car_accel_x
    car_y += car_accel_y

    # Fill the screen with black
    
    
    screen.blit(bg_image, (0, 0))



    rotated_car = pygame.transform.rotate(car_image, car_angle)
    rotated_car_rect = rotated_car.get_rect()
    rotated_car_rect.center = (car_x + car_width // 2 - rotation_offset * math.cos(math.radians(car_angle)),
                               car_y + car_height // 2 + rotation_offset * math.sin(math.radians(car_angle)))

    screen.blit(rotated_car, rotated_car_rect)


    points = []
    for point in [Point(car_rect.topleft[0] + car_x , car_rect.topleft[1] + car_y ), Point(car_rect.topright[0] + car_x, car_rect.topright[1] + car_y), Point(car_rect.bottomleft[0] + car_x, car_rect.bottomleft[1] + car_y), Point(car_rect.bottomright[0] + car_x, car_rect.bottomright[1] + car_y)]:
        points.append(point)

    point_topleft, point_topright, point_bottomleft, point_bottomright = rotate_points(*points, -car_angle, rotated_car_rect.center)
    car_bounding_box = [Line_Segment(point_bottomleft, point_bottomright), Line_Segment(point_topleft, point_topright), Line_Segment(point_bottomleft, point_topleft), Line_Segment(point_bottomright, point_topright)]
    colliding = False
    for line_bounding_box in car_bounding_box:
        for line in borders:
            colliding = colliding or line_bounding_box.doIntersect(line)
    if colliding:
        respawn_car()
    


    # Draw the background
    

    # Rotate the car image around the front wheels and draw it at the correct position
    

    # Check if the car touches the color #393939 (dark gray)
    

    pygame.display.flip()

    clock.tick(60)  # FPS limit
print(borders)

# Quit properly
pygame.quit()
