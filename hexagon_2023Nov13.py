# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 13:13:29 2023

@author: user
"""
# hexagon.py
# to check if we can detect a moving source's direction, with 6 detecting points.  
#
import numpy as np
from PIL import Image, ImageDraw, ImagePath, ImageFont
import math
import torch 


# polygon sides
def side_inside(n):
    s = n*(n-1)/2 - n
    return s

def side(n):
    s = n*(n-1)/2
    return s

# cylinder mass
def mass(r, l):
    a = r/2*r/2*np.pi
    mas = l*a*7.9
    return mas

# in meter
def kinetics(m, v):
    e = 1/2*m*v*v
    return e


# doppler effect with angle, vel_s is getting closer.
# assume v_sound > v_source.
def doppler_effect_angle(freq_source, theta, v_source):
    v_sound = 343
    v_source = v_source*np.cos(theta)
    freq = v_sound / (v_sound-v_source) * freq_source
    return freq

# source is moving toward the point.
# assume v_sound > v_source
def doppler_effect(freq_source, v_source):
    v_sound = 343
    freq_detected = freq_source * (v_sound) / (v_sound-v_source)
    return freq_detected

# count different directions of polygon's sides
def count_directions(points):
    n = points
    if n % 2 == 1:
        count = n * (n-1) / 2
    elif n % 2 == 0:
        count = n * (n-1) / 2
        count -= (n/2 - 1)*(n/2)
    return count

# convert freq to wavelength
def freq_to_wavelen(freq):
    v_sound = 343
    wavelen = v_sound / freq
    return wavelen


# a class of 6 points and 1 source.
class hexagon_and_source():
    def __init__(self, source):
        self.radius = 300                  # radius of hexagon
        self.origin = [0, 0]               # origin of hexagon
        self.points = []                   # hexagon points
        self.hexagon_rot = 0               # rotation from normal axis.
        self.sound_v = 343                 # sound velocity in m/sec
        self.source_v = 100                # source velocity in m/sec
        self.source_freq = 440             # source freq in hz.
        self.source = source               # positions of the source
        self.source_way = [-1, 0]          # source moving direction.

    # write 6 points of the hexagon.
    def write_6_points(self):
        theta = self.hexagon_rot
        r = self.radius
        for i in range(6):
            theta = theta + np.pi/3
            point = [round(r*np.cos(theta), 1), round(r*np.sin(theta), 1)]
            self.points.append(point)
        return

    # compute 6 frequency in Doppler effect.
    def compute_6_freq(self):        
        self.theta_cos = []
        for i in range(6):
            #source_vector = [self.source_way[0]*self.source_v, self.source_way[1]*self.source_v]
            src2pnt_vector = [self.points[i][0]-self.source[0], self.points[i][1]-self.source[1]]
            theta_cos = (self.source_way[0]*src2pnt_vector[0] + self.source_way[1]*src2pnt_vector[1]) / np.linalg.norm(src2pnt_vector)        
            self.theta_cos.append(theta_cos)  
        self.freq = []
        for i in range(6):
            freq = self.sound_v / (self.sound_v - self.source_v*self.theta_cos[i]) * self.source_freq
            self.freq.append(freq)   
        return
    
    # compute freq to wavelen, on the 6 points.
    def compute_6_wavelen(self):
        if self.freq == None:
            print("Please run compute_6_freq() first.")
            return
        self.wavelen = []
        for i in range(6):
            wavelen = freq_to_wavelen(self.freq[i])
            self.wavelen.append(wavelen)
        return 
    
    # list all Dij. Dij = Xj - Xi.
    # show every Dij[i][j] in freq, of 6 points.
    def Dij_vote(self, way):
        self.Dij = torch.zeros(6, 6)
        
        print("i\j       0      1      2      3      4      5  ")
        for i in range(6): 
            print(str(i)+ "    ", end="")
            for j in range(6):
                if way == "freq":
                    result = self.freq[j] - self.freq[i]  
                    result = round(result, 1)
                    self.Dij[i][j] = result
                if way == "wavelen":    
                    result = self.wavelen[j] - self.wavelen[i]  
                    result = round(result, 1)
                    self.Dij[i][j] = result
                
                if result > 0 :
                    print(" ", end="")
                if abs(result) == 0:
                    print("     0 ", end="")
                elif abs(result) < 10 :
                    print("  %.1f " % result, end="")
                elif abs(result) < 100 :
                    print(" %.1f " % result, end="")
                elif abs(result) < 1000 :
                    print("%.1f " % result, end="")
            print("")
        
        Dij = self.Dij    
        Dij = torch.tensor(Dij)
        Dij = torch.round(Dij, decimals=1)
        
        idx = torch.argmax(Dij)
        maximum = Dij.view(-1)[idx]     
        for i in range(6):
            for j in range(6):
                if Dij[i][j].item() == maximum:
                    idx, idx2 = i, j            
        print("The vote is from " + str(idx) + " to " + str(idx2))    
        return


# draw sources.
def draw_sources(hexagons):
    sources = []
    for hexagon in hexagons:
        sources.append(hexagon.source)
    hexagon = hexagons[0]
    
    shift_x, shift_y = (hexagon.radius+30), hexagon.radius
    xy = [tuple([hexagon.points[0][0]+shift_x, hexagon.points[0][1]+shift_y]), 
          tuple([hexagon.points[1][0]+shift_x, hexagon.points[1][1]+shift_y]), 
          tuple([hexagon.points[2][0]+shift_x, hexagon.points[2][1]+shift_y]), 
          tuple([hexagon.points[3][0]+shift_x, hexagon.points[3][1]+shift_y]), 
          tuple([hexagon.points[4][0]+shift_x, hexagon.points[4][1]+shift_y]), 
          tuple([hexagon.points[5][0]+shift_x, hexagon.points[5][1]+shift_y])]  
    size = [shift_x*2, shift_y*2]
    img = Image.new("RGB", size, "#f9f9f9")  
    img1 = ImageDraw.Draw(img)   
    img1.polygon(xy, fill ="#eeeeff", outline ="blue")  
    font = ImageFont.truetype(r"C:\Windows\Fonts\Arial.ttf", 20)
    for i in range(6):
        img1.ellipse(xy=[(hexagon.points[i][0]+shift_x-10, hexagon.points[i][1]+shift_y-10), 
                         (hexagon.points[i][0]+shift_x+10, hexagon.points[i][1]+shift_y+10)], 
                     fill="white", 
                     outline="red")
        img1.text(xy=(hexagon.points[i][0]+shift_x-5, hexagon.points[i][1]+shift_y-10), 
                  font=font,
                  fill="black", text=str(i), align="center")
    count=0
    for source in sources:
        count += 1
        img1.ellipse(xy=[(source[0]-15+shift_x, source[1]-15+shift_y), 
                         (source[0]+15+shift_x, source[1]+15+shift_y)], 
                     fill="yellow", outline="red") 
        img1.text((source[0]+shift_x-5, source[1]+shift_y-10), font=font, fill="black", 
                  text=str(count), align="center")
    img.show()
    img.save("output.jpg")
    return


# count Dij and get the predicted direction.
def Dij_count(hexagons, way):
    max_idx = []
    if way == "freq":
        print()
    elif way == "wavelen":       
        print()
    return max_idx


if __name__ == "__main__" :      
    y0 = 200
    
    hexagon_1 = hexagon_and_source([150, y0])
    hexagon_1.write_6_points()
    hexagon_1.compute_6_freq()
    
    hexagon_2 = hexagon_and_source([75, y0])
    hexagon_2.write_6_points()
    hexagon_2.compute_6_freq()
    
    hexagon_3 = hexagon_and_source([0, y0])
    hexagon_3.write_6_points()
    hexagon_3.compute_6_freq()
    
    hexagon_4 = hexagon_and_source([-75, y0])
    hexagon_4.write_6_points()
    hexagon_4.compute_6_freq()
    
    hexagon_5 = hexagon_and_source([-150, y0])
    hexagon_5.write_6_points()
    hexagon_5.compute_6_freq()

    hexagons = [hexagon_1, hexagon_2, hexagon_3, hexagon_4, hexagon_5]
    draw_sources(hexagons)
    
    print("")
    print("Dij = freq[j] - freq[i]")
    print("hexagon_1 Dij:")
    hexagon_1.Dij_vote("freq")
  
    print("")
    print("hexagon_2 Dij:")
    hexagon_2.Dij_vote("freq")
 
    print("")
    print("hexagon_3 Dij:")
    hexagon_3.Dij_vote("freq")

    print("")
    print("hexagon_4 Dij:")
    hexagon_4.Dij_vote("freq")
    
    print("")
    print("hexagon_5 Dij:")
    hexagon_5.Dij_vote("freq")
    print("")
    
    
    
    