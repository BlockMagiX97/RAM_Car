import math
import torch.multiprocessing as mp
import torch
import numpy as np

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
 
    # Method used to display X and Y coordinates
    # of a point
    def distance_to(self, other_point):
        return math.sqrt((self.x - other_point.x)**2 + (self.y - other_point.y)**2)
    
    def __repr__(self) -> str:
        return(f'Point({self.x}, {self.y})')

class Line_Segment:
    def __init__(self, start_point, end_point):
        self.A = start_point
        self.B = end_point
        self.has_collided_with = False

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

    def midpoint(self):
        x1, y1 = self.A.x, self.A.y
        x2, y2 = self.B.x, self.B.y

        midpoint_x = (x1 + x2) / 2
        midpoint_y = (y1 + y2) / 2

        return (Point(midpoint_x, midpoint_y))

def points_on_circle(center, diameter, num_points=10, angle_offset=0.0):
    points = []
    radius = diameter / 2
    angle_increment = 2 * math.pi / num_points

    for i in range(num_points):
        angle = i * angle_increment + angle_offset
        x = center.x + radius * math.cos(angle)
        y = center.y + radius * math.sin(angle)
        points.append(Point(x, y))

    return points

class HalfLine:
    def __init__(self, origin, direction_point):
        self.origin = origin
        self.direction_point = direction_point
        self.is_vertical = self.direction_point.x == self.origin.x
        self.slope, self.intercept = self.calculate_slope_and_intercept()

    def calculate_slope_and_intercept(self):
        if not self.is_vertical:
            slope = (self.direction_point.y - self.origin.y) / (self.direction_point.x - self.origin.x)
            if abs(slope) < 1e-9:  # Check for slope close to zero
                slope = 1e-9
            intercept = self.origin.y - slope * self.origin.x
            return slope, intercept
        else:
            # Handle the vertical half-line case (infinite slope)
            return float("inf"), None

    def find_intersection(self, segment):
        x1, y1 = self.origin.x, self.origin.y
        x2, y2 = self.direction_point.x, self.direction_point.y
        x3, y3 = segment.A.x, segment.A.y
        x4, y4 = segment.B.x, segment.B.y

        denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

        if denominator == 0:
            return None  # Lines are parallel or coincident

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denominator
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denominator

        if 0 <= t <= 1 and 0 <= u <= 1:
            intersection_x = x1 + t * (x2 - x1)
            intersection_y = y1 + t * (y2 - y1)
            return Point(intersection_x, intersection_y)

        return None  # Lines do not intersect within the given segments

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