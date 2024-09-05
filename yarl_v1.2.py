#################################################################################
#                                                                               #
# YARL is an endless roguelike game, written in Python3 plus Pygame library     #
# Copyright (C) 2024 by JSettler @ GitHub                                       #
#                                                                               #
# This program is free software: you can redistribute it and/or modify          #
# it under the terms of the GNU [Affero] General Public License as published by #
# the Free Software Foundation, either version 3 of the License, or             #
# (at your option) any later version.                                           #
#                                                                               #
# This program is distributed in the hope that it will be useful,               #
# but WITHOUT ANY WARRANTY; without even the implied warranty of                #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                 #
# GNU [Affero] General Public License for more details.                         #
#                                                                               #
# You should have received a copy of the GNU [Affero] General Public License    #
# along with this program.  If not, see <https://www.gnu.org/licenses/>.        #
#                                                                               #
# YARL may only be distributed together with its LICENSE and ReadMe files.      #
#                                                                               #
#################################################################################


import pygame
import sys
import random
import time
import heapq
import json
import os
import lz4.block
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple, Dict


# Constants
SCREEN_WIDTH = 1680
SCREEN_HEIGHT = 1050
TILE_SIZE = 20
FPS = 60
LEVEL_WIDTH = 249  # 83 * 3
LEVEL_HEIGHT = 147  # 49 * 3
MOVE_REPEAT_DELAY = 200
MOVE_REPEAT_INTERVAL = 1000 // 10  # Changed from 1000 // 5

# Colors
BLACK = (0, 0, 0)
DARK_GREY = (50, 50, 50)
WHITE = (255, 255, 255)
DARK_BLUE = (0, 0, 128)
GREY = (128, 128, 128)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
BRIGHT_ORANGE = (255, 165, 0)
DARK_RED = (50, 0, 0)
BLUISH_GREY = (30, 35, 50)


# Tile types
FLOOR = 0
WALL = 1
STAIRS_DOWN = 2
STAIRS_UP = 3
TRAP = 4
REVEALED_TRAP = 5
CHEST = 6
PLACED_ROCK = 7


PASS_REPEAT_DELAY = 400     # 0.4 seconds in milliseconds
PASS_REPEAT_INTERVAL = 20   # 50 would be 0.05 seconds (20 times per second) in milliseconds


def generate_tone(frequency, duration, volume=0.1, decay=1.0):
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    tone = np.sin(frequency * t * 2 * np.pi)
    envelope = np.exp(-t * decay)
    tone = (volume * tone * envelope * 32767).astype(np.int16)
    stereo_tone = np.column_stack((tone, tone))
    return pygame.sndarray.make_sound(stereo_tone)

def generate_heal_tick():
    return generate_tone(1000, 0.02, volume=0.15, decay=50)

def generate_tiny_heal_tick():
    return generate_tone(1000, 0.002, volume=0.25, decay=5)


def generate_death_sound():
    """
    sample_rate = 44100
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    tone1 = np.sin(440 * t * 2 * np.pi) * np.exp(-t * 2)
    tone2 = np.sin(220 * t * 2 * np.pi) * np.exp(-t * 1)
    tone3 = np.sin(110 * t * 2 * np.pi) * np.exp(-t * 0.5)
    
    combined = (tone1 + tone2 + tone3) / 3
    # Increase the volume by multiplying by a factor (e.g., 2.0 for twice as loud)
    combined *= 2.0  
    sound = (np.clip(combined, -1, 1) * 32767).astype(np.int16)
    stereo_sound = np.column_stack((sound, sound))
    return pygame.sndarray.make_sound(stereo_sound)
    """
    sample_rate = 44100
    duration = 1.5
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # A descending tone
    freq = 880 * 2**(-t*2)
    tone = np.sin(2 * np.pi * freq * t)
    
    # Add some noise for a more dramatic effect
    noise = np.random.rand(len(t))
    
    combined = (tone * 0.7 + noise * 0.3) * np.exp(-t * 3)
    sound = (np.clip(combined, -1, 1) * 32767).astype(np.int16)
    return pygame.sndarray.make_sound(np.column_stack((sound, sound)))



def generate_new_level_fanfare():
    sample_rate = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    tone1 = np.sin(440 * t * 2 * np.pi) * np.exp(-t * 3)
    tone2 = np.sin(550 * t * 2 * np.pi) * np.exp(-t * 3)
    tone3 = np.sin(660 * t * 2 * np.pi) * np.exp(-t * 3)
    
    fanfare = np.zeros_like(t)
    fanfare[:len(t)//3] = tone1[:len(t)//3]
    fanfare[len(t)//3:2*len(t)//3] = tone2[len(t)//3:2*len(t)//3]
    fanfare[2*len(t)//3:] = tone3[2*len(t)//3:]
    
    sound = (fanfare * 32767).astype(np.int16)
    stereo_sound = np.column_stack((sound, sound))  # Create a stereo sound
    return pygame.sndarray.make_sound(stereo_sound)


def generate_trapdoor_fall(volume=1.0):
    sample_rate = 44100
    duration = 1.5
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    freq = 400 * 2**(-(t*4) % 1)  # Descending pitch
    tone = np.sin(freq * t * 2 * np.pi)
    
    rumble = np.random.rand(len(t)) * np.exp(-t * 2)
    
    combined = (tone * 0.7 + rumble * 0.3) * np.exp(-t * 1.5)
    sound = (combined * volume * 32767).astype(np.int16)
    stereo_sound = np.column_stack((sound, sound))  # Create a stereo sound
    return pygame.sndarray.make_sound(stereo_sound)


def generate_chest_open_sound():
    sample_rate = 44100
    duration = 1.5
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # Creaking sound (frequency modulation)
    creak_freq = 1000 + 500 * np.sin(2 * np.pi * 2 * t)
    creak = np.sin(2 * np.pi * creak_freq * t) * np.exp(-t * 3)
    
    # Wooden thud
    thud_freq = 100
    thud = np.sin(2 * np.pi * thud_freq * t) * np.exp(-t * 20)
    
    # Metal clink
    clink_freq = 3000
    clink = np.sin(2 * np.pi * clink_freq * t) * np.exp(-t * 30)
    
    # Combine sounds
    sound = (creak * 0.7 + thud * 0.2 + clink * 0.1) * 0.5
    sound = (sound * 32767).astype(np.int16)
    
    return pygame.sndarray.make_sound(np.column_stack((sound, sound)))


def adjust_volume(self, volume):
    for sound in sound_manager.sounds.values():
        sound.set_volume(volume)
    # print(f"Volume set to {volume}")

"""
# Create the sounds
move_sound = generate_tone(440, 0.01, volume=0.02)
heal_tick_sound = generate_heal_tick()
attack_sound = generate_tone(880, 0.05, volume=0.07)
hit_sound = generate_tone(220, 0.3)
death_sound = generate_death_sound()
new_level_sound = generate_new_level_fanfare()
trapdoor_sound = generate_trapdoor_fall()

#-----------------------------------------------------------------------------

def heuristic(a, b):
    return abs(b[0] - a[0]) + abs(b[1] - a[1])

def astar(start, goal, grid):
    neighbors = [(0,1),(0,-1),(1,0),(-1,0)]
    close_set = set()
    came_from = {}
    gscore = {start:0}
    fscore = {start:heuristic(start, goal)}
    open_set = []
    heapq.heappush(open_set, (fscore[start], start))
    
    while open_set:
        current = heapq.heappop(open_set)[1]
        
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path

        close_set.add(current)
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j
            tentative_g_score = gscore[current] + 1
            if 0 <= neighbor[0] < len(grid) and 0 <= neighbor[1] < len(grid[0]):
                if grid[neighbor[1]][neighbor[0]] == WALL:
                    continue
            else:
                continue
                
            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                continue
                
            if  tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1]for i in open_set]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = gscore[neighbor] + heuristic(neighbor, goal)
                heapq.heappush(open_set, (fscore[neighbor], neighbor))
                
    return False
"""

# Global sound manager
class SoundManager:
    def __init__(self):
        self.enabled = True
        self.volume = 0.3  # Start at 50% volume
        self.sound_data = {
            'move': generate_tone(440, 0.005),
            'heal_tick': generate_heal_tick(),
            'tiny_heal_tick': generate_tiny_heal_tick(),
            'attack': generate_tone(880, 0.05),
            'player_hit': generate_tone(220, 0.3),
            'monster_death': generate_tone(660, 0.2),
            'death': generate_death_sound(),
            'new_level': generate_new_level_fanfare(),
            'trapdoor': generate_trapdoor_fall(),
            'chest_open': generate_chest_open_sound(),
            'collect_rock': generate_tone(550, 0.1),
            'place_rock': generate_tone(330, 0.1),
        }
        self.sounds = {}
        self.create_sound_objects()


    def create_sound_objects(self):
        for name, data in self.sound_data.items():
            if isinstance(data, pygame.mixer.Sound):
                sound = data
            elif isinstance(data, np.ndarray):
                if len(data.shape) == 1:  # Mono sound
                    stereo_data = np.column_stack((data, data))
                else:  # Already stereo
                    stereo_data = data
                sound = pygame.sndarray.make_sound(stereo_data)
            else:
                raise ValueError(f"Unsupported sound data type for {name}")
            
            sound.set_volume(self.volume)
            self.sounds[name] = sound




    def play(self, sound_name):
        if self.enabled and sound_name in self.sounds:
            # print(f"Playing sound: {sound_name}")  # Debug print
            self.sounds[sound_name].play()


    def toggle(self):
        self.enabled = not self.enabled
        return self.enabled

    def set_volume(self, sound_name, volume):
        if sound_name in self.sounds:
            self.sounds[sound_name].set_volume(volume)

    def adjust_volume(self, change):
        self.volume = max(0.0, min(1.0, self.volume + change))
        self.create_sound_objects()  # Recreate all sound objects with new volume
        return self.volume


    def apply_volume(self):
        for sound in self.sounds.values():
            sound.set_volume(self.volume)




pygame.init()
try:
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
except pygame.error:
    print("Warning: Unable to initialize sound system")
    sound_manager.enabled = False

# Create a global instance of SoundManager
sound_manager = SoundManager()


# New Item classes
class Item:
    def __init__(self, name):
        self.name = name

class Sword(Item):
    def __init__(self, level):
        super().__init__(f"Level {level} Sword")
        self.level = level
        self.damage = 10 + (level - 1) * 2

class Pickaxe(Item):
    def __init__(self, durability):
        super().__init__("Pickaxe")
        self.durability = durability

class Rocks(Item):
    def __init__(self, amount):
        super().__init__("Rocks")
        self.amount = amount


# New Inventory class
class Inventory:
    def __init__(self):
        self.items = []
        self.active_item = None

    def add_item(self, item):
        if isinstance(item, Pickaxe) and self.has_pickaxe():
            self.combine_pickaxes(item)
        else:
            self.items.append(item)
        if self.active_item is None:
            self.active_item = item

    """
    def switch_item(self):
        if len(self.items) > 1:
            current_index = self.items.index(self.active_item)
            next_index = (current_index + 1) % len(self.items)
            self.active_item = self.items[next_index]
    """

    def has_pickaxe(self):
        return any(isinstance(item, Pickaxe) for item in self.items)

    def combine_pickaxes(self, new_pickaxe):
        for item in self.items:
            if isinstance(item, Pickaxe):
                item.durability += new_pickaxe.durability
                return

    def get_rocks(self):
        for item in self.items:
            if isinstance(item, Rocks):
                return item
        return None



@dataclass
class GameObject:
    x: int
    y: int
    color: Tuple[int, int, int]
    symbol: str

@dataclass
class Weapon:
    name: str
    damage: int

@dataclass
class Monster:
    x: int
    y: int
    max_hp: int
    current_hp: int
    monster_type: str = "regular"  # "regular" or "sentinel"
    damage: int = 0
    aggro_range: int = 16
    pursuit_range: int = 16
    territory_center: Tuple[int, int] = None
    territory_radius: int = 10
    direction: str = "right"  # "right", "left", "up", "down"
    path: List[Tuple[int, int]] = field(default_factory=list)
    last_path_update: int = 0

    @property
    def size_factor(self):
        return max(0.05, min(1.0, self.current_hp / self.max_hp))


class Level:
    def __init__(self, level_number: int, seed: int = None, entry_type: str = 'stairs'):
        self.level_number = level_number
        self.seed = seed if seed is not None else random.randint(0, 2**32 - 1)
        self.grid = [[WALL for _ in range(LEVEL_WIDTH)] for _ in range(LEVEL_HEIGHT)]
        self.random = random.Random(self.seed)
        self.stairs_up_position = None
        self.stairs_down_position = None
        self.revealed_traps: Dict[Tuple[int, int], bool] = {}
        self.healed_locations = set()
        self.chest_position = None
        self.chest_opened = False
        self.first_chest_found = False
        self.placed_rocks = set()  # To keep track of placed rocks
        self.monsters: List[Monster] = []
        self.monster_damage = 10 + (level_number - 1)
        self.cleared = False
        self.connected_areas = []
        self.entry_type = entry_type  # 'stairs' or 'trapdoor'
        self.trapdoor_entry_position = None

        self.initial_monster_count = 0
        self.placed_rocks_count = 0
        self.has_mined = False

        self.fog_of_war = [[False for _ in range(LEVEL_WIDTH)] for _ in range(LEVEL_HEIGHT)]
        self.generate()
        self.sentinel_count = max(0, (level_number - 6) // 6)
        self.generate_monsters()



    def to_dict(self, is_deepest_level):
        data = {
            'level_number': self.level_number,
            'seed': self.seed,
            'grid': self.grid,
            'stairs_up_position': self.stairs_up_position,
            'stairs_down_position': self.stairs_down_position,
            'revealed_traps': list(self.revealed_traps.keys()),
            'monsters': [{'x': m.x, 'y': m.y, 'max_hp': m.max_hp, 'current_hp': m.current_hp,
                          'monster_type': m.monster_type, 'damage': m.damage,
                          'aggro_range': m.aggro_range, 'pursuit_range': m.pursuit_range,
                          'territory_center': m.territory_center, 'territory_radius': m.territory_radius,
                          'direction': m.direction} for m in self.monsters],
            'sentinel_count': self.sentinel_count,
            'fog_of_war': self.fog_of_war,
            'chest_position': self.chest_position,
            'chest_opened': self.chest_opened,
            'cleared': self.cleared,
            'has_mined': self.has_mined if is_deepest_level else False ,
            'placed_rocks': list(self.placed_rocks)
        }
        if is_deepest_level:
            data['healed_locations'] = list(self.healed_locations)
        return data


    @classmethod
    def from_dict(cls, data, is_deepest_level):
        level = cls(data['level_number'], data['seed'])
        level.grid = data['grid']
        level.stairs_up_position = data['stairs_up_position']
        level.stairs_down_position = data['stairs_down_position']
        level.revealed_traps = {tuple(pos): True for pos in data['revealed_traps']}
        level.monsters = [Monster(**m) for m in data['monsters']]
        level.sentinel_count = data.get('sentinel_count', 0)
        level.fog_of_war = data['fog_of_war']
        level.chest_position = data.get('chest_position')
        level.chest_opened = data.get('chest_opened', False)
        level.cleared = data.get('cleared', False)
        if is_deepest_level:
            level.has_mined = data.get('has_mined', False)
        level.placed_rocks = set(tuple(pos) for pos in data['placed_rocks'])
        if is_deepest_level and 'healed_locations' in data:
            level.healed_locations = set(tuple(loc) for loc in data['healed_locations'])
        return level


    def generate(self):
        # Generate the central part (original size)
        central_width = LEVEL_WIDTH // 3
        central_height = LEVEL_HEIGHT // 3
        for y in range(central_height):
            for x in range(central_width):
                if self.random.random() < 0.7:  # Adjust this value to change dungeon density
                    self.grid[y + central_height][x + central_width] = FLOOR

        # Expand the dungeon to the surrounding areas
        for _ in range(LEVEL_WIDTH * LEVEL_HEIGHT // 5):  # Adjust this value to change expansion amount
            x = self.random.randint(0, LEVEL_WIDTH - 1)
            y = self.random.randint(0, LEVEL_HEIGHT - 1)
            if self.grid[y][x] == FLOOR:
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < LEVEL_WIDTH and 0 <= ny < LEVEL_HEIGHT:
                        self.grid[ny][nx] = FLOOR

        # Add some random rooms
        num_rooms = random.randint(5, 10)
        for _ in range(num_rooms):
            room_width = random.randint(5, 15)
            room_height = random.randint(5, 15)
            room_x = random.randint(0, LEVEL_WIDTH - room_width - 1)
            room_y = random.randint(0, LEVEL_HEIGHT - room_height - 1)
            
            for y in range(room_y, room_y + room_height):
                for x in range(room_x, room_x + room_width):
                    self.grid[y][x] = FLOOR

        # Connect rooms with corridors
        self.connect_rooms()

        # Place stairs and other elements
        empty_tiles = [(x, y) for y in range(LEVEL_HEIGHT) for x in range(LEVEL_WIDTH) if self.grid[y][x] == FLOOR]
        
        if self.level_number > 1:
            if empty_tiles:
                stairs_up = self.random.choice(empty_tiles)
                self.grid[stairs_up[1]][stairs_up[0]] = STAIRS_UP
                self.stairs_up_position = stairs_up
                empty_tiles.remove(stairs_up)
            else:
                # If no empty tiles, create stairs in the center
                center_x, center_y = LEVEL_WIDTH // 2, LEVEL_HEIGHT // 2
                self.grid[center_y][center_x] = STAIRS_UP
                self.stairs_up_position = (center_x, center_y)

        if empty_tiles:
            stairs_down = self.random.choice(empty_tiles)
            self.grid[stairs_down[1]][stairs_down[0]] = STAIRS_DOWN
            self.stairs_down_position = stairs_down
            empty_tiles.remove(stairs_down)
        else:
            # If no empty tiles, create stairs in the center
            center_x, center_y = LEVEL_WIDTH // 2, LEVEL_HEIGHT // 2
            if self.grid[center_y][center_x] != STAIRS_UP:
                self.grid[center_y][center_x] = STAIRS_DOWN
                self.stairs_down_position = (center_x, center_y)
            else:
                # If center is occupied by up stairs, place down stairs adjacent
                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    new_x, new_y = center_x + dx, center_y + dy
                    if 0 <= new_x < LEVEL_WIDTH and 0 <= new_y < LEVEL_HEIGHT:
                        self.grid[new_y][new_x] = STAIRS_DOWN
                        self.stairs_down_position = (new_x, new_y)
                        break

        # Place more traps
        num_traps = self.random.randint(7, 15)  # Increased number of traps, from 3-15 to 7-15
        for _ in range(num_traps):
            if empty_tiles:
                trap = self.random.choice(empty_tiles)
                self.grid[trap[1]][trap[0]] = TRAP
                empty_tiles.remove(trap)

        # Update chest generation
        if self.level_number >= 6:
            chest_probability = 0.7 if not self.first_chest_found else 0.4
            if random.random() < chest_probability:
                empty_tiles = [(x, y) for y in range(LEVEL_HEIGHT) for x in range(LEVEL_WIDTH) if self.grid[y][x] == FLOOR]
                if empty_tiles:
                    chest_pos = random.choice(empty_tiles)
                    self.grid[chest_pos[1]][chest_pos[0]] = CHEST
                    self.chest_position = chest_pos

        self.ensure_connectivity()
        if self.entry_type == 'stairs':
            self.place_stairs()
        else:  # trapdoor
            self.place_trapdoor_entry()


    def place_stairs(self):
        # Place stairs up (if not the first level)
        if self.level_number > 0:
            start_pos = random.choice(self.connected_areas[0])
            self.stairs_up_position = start_pos
            self.grid[start_pos[1]][start_pos[0]] = STAIRS_UP

        # Place stairs down
        end_pos = random.choice(self.connected_areas[0])
        while end_pos == self.stairs_up_position:
            end_pos = random.choice(self.connected_areas[0])
        self.stairs_down_position = end_pos
        self.grid[end_pos[1]][end_pos[0]] = STAIRS_DOWN


    def place_trapdoor_entry(self):
        entry_pos = random.choice(self.connected_areas[0])
        self.trapdoor_entry_position = entry_pos
        # We don't change the grid here, as the trapdoor entry should be on a normal floor tile



    def ensure_connectivity(self):
        visited = [[False for _ in range(LEVEL_WIDTH)] for _ in range(LEVEL_HEIGHT)]
        self.connected_areas = []

        for y in range(LEVEL_HEIGHT):
            for x in range(LEVEL_WIDTH):
                if self.grid[y][x] == FLOOR and not visited[y][x]:
                    area = self.flood_fill(x, y, visited)
                    if area:  # Only add non-empty areas
                        self.connected_areas.append(area)

        # Sort areas by size, largest first
        self.connected_areas.sort(key=len, reverse=True)

        # Connect all areas to the largest area
        for i in range(1, len(self.connected_areas)):
            self.connect_areas(self.connected_areas[0], self.connected_areas[i])



    def flood_fill(self, start_x, start_y, visited):
        area = []
        queue = deque([(start_x, start_y)])
        
        while queue:
            x, y = queue.popleft()
            if not (0 <= x < LEVEL_WIDTH and 0 <= y < LEVEL_HEIGHT) or visited[y][x] or self.grid[y][x] != FLOOR:
                continue

            visited[y][x] = True
            area.append((x, y))

            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                queue.append((x + dx, y + dy))

        return area



    def connect_areas(self, area1, area2):
        start = random.choice(area1)
        end = random.choice(area2)

        x, y = start
        while (x, y) != end:
            if x < end[0]:
                x += 1
            elif x > end[0]:
                x -= 1
            elif y < end[1]:
                y += 1
            elif y > end[1]:
                y -= 1

            self.grid[y][x] = FLOOR


    def place_player_start(self):
        # Choose a random position in the largest connected area
        start_pos = random.choice(self.connected_areas[0])
        self.stairs_up_position = start_pos
        self.grid[start_pos[1]][start_pos[0]] = STAIRS_UP


    def find_spawn_location(self):
        if self.entry_type == 'stairs':
            return self.stairs_up_position
        else:  # trapdoor
            return self.trapdoor_entry_position



    def generate_monsters(self):
        num_monsters = self.random.randint(18, 48)
        base_hp = 50
        base_damage = 10 + (self.level_number - 1)  # This was the original scaling for damage

        # Calculate level-based increases
        level_hp_increase = max(0, self.level_number - 1) * 5 // 100 * base_hp
        monster_hp = base_hp + level_hp_increase
        monster_damage = base_damage

        # Generate regular monsters
        empty_tiles = [(x, y) for y in range(LEVEL_HEIGHT) for x in range(LEVEL_WIDTH) if self.grid[y][x] == FLOOR]
        for _ in range(num_monsters):
            if empty_tiles:
                monster_pos = self.random.choice(empty_tiles)
                self.monsters.append(Monster(monster_pos[0], monster_pos[1], monster_hp, monster_hp, "regular", monster_damage))
                empty_tiles.remove(monster_pos)

        # Generate Sentinels
        for _ in range(self.sentinel_count):
            sentinel_hp = monster_hp * 4
            sentinel_damage = monster_damage * 2
            x, y = self.find_empty_space()
            territory_radius = self.random.randint(4, 13)  # Random radius between 4 and 13
            self.monsters.append(Monster(x, y, sentinel_hp, sentinel_hp, "sentinel", sentinel_damage, 16, 32, (x, y), territory_radius))

        self.initial_monster_count = len(self.monsters)



    def find_empty_space(self):
        while True:
            x = self.random.randint(0, LEVEL_WIDTH - 1)
            y = self.random.randint(0, LEVEL_HEIGHT - 1)
            if self.grid[y][x] == FLOOR and not any(m.x == x and m.y == y for m in self.monsters):
                return x, y



    def connect_rooms(self):
        def find_floor_tile():
            while True:
                x = random.randint(0, LEVEL_WIDTH - 1)
                y = random.randint(0, LEVEL_HEIGHT - 1)
                if self.grid[y][x] == FLOOR:
                    return x, y

        for _ in range(50):  # Adjust this number to change corridor density
            start_x, start_y = find_floor_tile()
            end_x, end_y = find_floor_tile()
            
            x, y = start_x, start_y
            while (x, y) != (end_x, end_y):
                if random.random() < 0.5:
                    x += 1 if x < end_x else -1
                else:
                    y += 1 if y < end_y else -1
                self.grid[y][x] = FLOOR

    def find_spawn_location(self, min_distance=5):
        while True:
            x = self.random.randint(0, LEVEL_WIDTH - 1)
            y = self.random.randint(0, LEVEL_HEIGHT - 1)
            if self.grid[y][x] == FLOOR:
                if all(max(abs(m.x - x), abs(m.y - y)) >= min_distance for m in self.monsters):
                    return x, y

    def get_monster_count(self):
        return len(self.monsters)

    def reveal_trap(self, x: int, y: int):
        self.revealed_traps[(x, y)] = True



    def is_valid_move(self, x, y):
        if not (0 <= x < LEVEL_WIDTH and 0 <= y < LEVEL_HEIGHT):
            return False
        if self.grid[y][x] in [WALL, PLACED_ROCK, CHEST]:
            return False
        if self.is_tile_occupied_by_monster(x, y):
            return False
        # Remove REVEALED_TRAP from the check, as we want to allow movement onto trap tiles
        return True


    def is_valid_move_for_mob(self, x, y):
        if not (0 <= x < LEVEL_WIDTH and 0 <= y < LEVEL_HEIGHT):
            return False
        if self.grid[y][x] in [WALL, PLACED_ROCK, CHEST, REVEALED_TRAP, STAIRS_UP, STAIRS_DOWN]:
            return False
        if self.is_tile_occupied_by_monster(x, y):
            return False
        return True


    def move_monsters(self, player_x, player_y):
        attacking_monsters = []
        for monster in self.monsters:
            dx = player_x - monster.x
            dy = player_y - monster.y
            distance = max(abs(dx), abs(dy))  # Chebyshev distance

            if distance <= 1:  # Monster is adjacent to player
                attacking_monsters.append(monster)
            elif monster.monster_type == "regular":
                self.move_regular_monster(monster, player_x, player_y, dx, dy, distance)
            else:  # sentinel
                self.move_sentinel_monster(monster, player_x, player_y, dx, dy, distance)

        return attacking_monsters



    def move_regular_monster(self, monster, player_x, player_y, dx, dy, distance):
        monster.aggro_range = 28  # Increase pursuit distance
        current_time = pygame.time.get_ticks()

        if distance <= monster.aggro_range:
            if distance < 21 and (current_time - monster.last_path_update > 400 or not monster.path):  # Update path every 0.4 seconds
                # Use A* algorithm
                start = (monster.x, monster.y)
                goal = (player_x, player_y)
                # path = self.astar(start, goal, max_depth=20)  # Limit search depth
                # path = self.astar(start, goal, self.grid)  # Use the original astar method
                path = self.astar(start, goal, max_depth=20)
                if path:
                    monster.path = path[1:]  # Exclude start position
                    monster.last_path_update = current_time

            if monster.path:
                next_pos = monster.path[0]
                if self.is_valid_move_for_mob(next_pos[0], next_pos[1]):
                    monster.x, monster.y = next_pos
                    monster.path.pop(0)
                    monster.direction = self.get_direction(next_pos[0] - monster.x, next_pos[1] - monster.y)
                else:
                    monster.path.clear()  # Clear path if blocked
            else:
                # Simple pursuit logic
                self.simple_pursuit(monster, dx, dy)
        else:
            # Random movement
            self.random_movement(monster)

    #-----------------------------------------------------------------------------------------------------------------------------------------------

    def simple_pursuit(self, monster, dx, dy):
        if abs(dx) > abs(dy):
            new_x = monster.x + (1 if dx > 0 else -1)
            if self.is_valid_move_for_mob(new_x, monster.y):
                monster.x = new_x
                monster.direction = "right" if dx > 0 else "left"
        else:
            new_y = monster.y + (1 if dy > 0 else -1)
            if self.is_valid_move_for_mob(monster.x, new_y):
                monster.y = new_y
                monster.direction = "down" if dy > 0 else "up"



    def random_movement(self, monster):
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        random.shuffle(directions)
        for dx, dy in directions:
            new_x, new_y = monster.x + dx, monster.y + dy
            if self.is_valid_move_for_mob(new_x, new_y):
                monster.x, monster.y = new_x, new_y
                monster.direction = self.get_direction(dx, dy)
                break

    def get_direction(self, dx, dy):
        if dx > 0:
            return "right"
        elif dx < 0:
            return "left"
        elif dy > 0:
            return "down"
        else:
            return "up"

    def is_tile_occupied_by_monster(self, x, y):
        return any(monster.x == x and monster.y == y for monster in self.monsters)

    """
    def heuristic(self, a, b):
        return abs(b[0] - a[0]) + abs(b[1] - a[1])

    def astar(self, start, goal, grid):
        neighbors = [(0,1),(0,-1),(1,0),(-1,0)]
        close_set = set()
        came_from = {}
        gscore = {start:0}
        fscore = {start:self.heuristic(start, goal)}
        open_set = []
        heapq.heappush(open_set, (fscore[start], start))
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path

            close_set.add(current)
            for i, j in neighbors:
                neighbor = current[0] + i, current[1] + j
                tentative_g_score = gscore[current] + 1
                if 0 <= neighbor[0] < len(grid) and 0 <= neighbor[1] < len(grid[0]):
                    if grid[neighbor[1]][neighbor[0]] == WALL:
                        continue
                else:
                    continue
                    
                if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                    continue
                    
                if  tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1]for i in open_set]:
                    came_from[neighbor] = current
                    gscore[neighbor] = tentative_g_score
                    fscore[neighbor] = gscore[neighbor] + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (fscore[neighbor], neighbor))
                    
        return False
    """

    #--------------------------------------------------
    """
    def astar(self, start, goal, max_depth=20):
        def heuristic(a, b):
            return abs(b[0] - a[0]) + abs(b[1] - a[1])

        neighbors = [(0,1),(0,-1),(1,0),(-1,0)]
        close_set = set()
        came_from = {}
        gscore = {start: 0}
        fscore = {start: heuristic(start, goal)}
        open_heap = [(fscore[start], start)]

        while open_heap:
            current = heapq.heappop(open_heap)[1]

            if current == goal or gscore[current] >= max_depth:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            close_set.add(current)

            for i, j in neighbors:
                neighbor = current[0] + i, current[1] + j
                if not self.is_valid_move_for_mob(neighbor[0], neighbor[1]):
                    continue

    """

    """
    def astar(self, start, goal, grid, max_depth=20):
        def heuristic(a, b):
            return abs(b[0] - a[0]) + abs(b[1] - a[1])

        neighbors = [(0,1),(0,-1),(1,0),(-1,0)]
        close_set = set()
        came_from = {}
        gscore = {start: 0}
        fscore = {start: heuristic(start, goal)}
        open_heap = [(fscore[start], start)]

        while open_heap:
            current = heapq.heappop(open_heap)[1]

            if current == goal or gscore[current] >= max_depth:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            close_set.add(current)

            for i, j in neighbors:
                neighbor = current[0] + i, current[1] + j
                if not self.is_valid_move_for_mob(neighbor[0], neighbor[1]):
                    continue

                tentative_g_score = gscore[current] + 1

                if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                    continue

                if tentative_g_score < gscore.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    gscore[neighbor] = tentative_g_score
                    fscore[neighbor] = gscore[neighbor] + heuristic(neighbor, goal)
                    heapq.heappush(open_heap, (fscore[neighbor], neighbor))

        return None
    """

    def astar(self, start, goal, max_depth=20):
        def heuristic(a, b):
            return abs(b[0] - a[0]) + abs(b[1] - a[1])

        neighbors = [(0,1),(0,-1),(1,0),(-1,0)]
        close_set = set()
        came_from = {}
        gscore = {start: 0}
        fscore = {start: heuristic(start, goal)}
        open_heap = [(fscore[start], start)]

        while open_heap:
            current = heapq.heappop(open_heap)[1]

            if current == goal or gscore[current] >= max_depth:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            close_set.add(current)

            for i, j in neighbors:
                neighbor = current[0] + i, current[1] + j
                if not self.is_valid_move_for_mob(neighbor[0], neighbor[1]):
                    continue

                tentative_g_score = gscore[current] + 1

                if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                    continue

                if tentative_g_score < gscore.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    gscore[neighbor] = tentative_g_score
                    fscore[neighbor] = gscore[neighbor] + heuristic(neighbor, goal)
                    heapq.heappush(open_heap, (fscore[neighbor], neighbor))

        return None


    def player_astar(self, start, goal, max_depth=100, timeout=1.0):
        def heuristic(a, b):
            return abs(b[0] - a[0]) + abs(b[1] - a[1])

        neighbors = [(0,1),(0,-1),(1,0),(-1,0)]
        close_set = set()
        came_from = {}
        gscore = {start: 0}
        fscore = {start: heuristic(start, goal)}
        open_set = [(fscore[start], start)]

        start_time = time.time()

        while open_set:
            if time.time() - start_time > timeout:
                # print("Pathfinding timed out")  # Debug print
                return None

            current = heapq.heappop(open_set)[1]

            if current == goal or (abs(current[0] - goal[0]) <= 1 and abs(current[1] - goal[1]) <= 1):
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            if gscore[current] >= max_depth:
                continue

            close_set.add(current)

            for i, j in neighbors:
                neighbor = current[0] + i, current[1] + j
                if not self.is_valid_move(neighbor[0], neighbor[1]):
                    continue

                tentative_g_score = gscore[current] + 1

                if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                    continue

                if tentative_g_score < gscore.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    gscore[neighbor] = tentative_g_score
                    fscore[neighbor] = gscore[neighbor] + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (fscore[neighbor], neighbor))

        # print("No path found")  # Debug print
        return None



    def move_sentinel_monster(self, monster, player_x, player_y, dx, dy, distance):
        if not hasattr(monster, 'random_move_counter'):
            monster.random_move_counter = 0

        if monster.random_move_counter > 0:
            # Random movement
            directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            random.shuffle(directions)
            for move_x, move_y in directions:
                new_x, new_y = monster.x + move_x, monster.y + move_y
                if self.is_valid_move_for_mob(new_x, new_y):
                    monster.x, monster.y = new_x, new_y
                    monster.direction = self.get_direction(move_x, move_y)
                    break
            monster.random_move_counter -= 1
        elif distance <= monster.aggro_range:
            # Pursuit mode
            new_x, new_y = monster.x, monster.y
            if abs(dx) > abs(dy):
                new_x += 1 if dx > 0 else -1
                monster.direction = "right" if dx > 0 else "left"
            else:
                new_y += 1 if dy > 0 else -1
                monster.direction = "down" if dy > 0 else "up"
            
            if self.is_valid_move_for_mob(new_x, new_y):
                monster.x, monster.y = new_x, new_y
            else:
                # Hit an obstacle, switch to random movement
                monster.random_move_counter = random.randint(2, 4)
        elif distance > monster.pursuit_range:
            # Return to territory
            tx, ty = monster.territory_center
            if abs(monster.x - tx) > abs(monster.y - ty):
                move_x = 1 if tx > monster.x else -1
                new_x, new_y = monster.x + move_x, monster.y
                monster.direction = "right" if move_x > 0 else "left"
            else:
                move_y = 1 if ty > monster.y else -1
                new_x, new_y = monster.x, monster.y + move_y
                monster.direction = "down" if move_y > 0 else "up"
            
            if self.is_valid_move_for_mob(new_x, new_y):
                monster.x, monster.y = new_x, new_y



    def is_blocked(self, x, y):
        if not (0 <= x < LEVEL_WIDTH and 0 <= y < LEVEL_HEIGHT):
            return True
        if self.grid[y][x] == WALL:
            return True
        if any(m for m in self.monsters if m.x == x and m.y == y):
            return True
        return False



    def dig_tunnels(self, x, y):
        def find_nearest_open(start_x, start_y, dx, dy, max_length=50):
            cx, cy = start_x, start_y
            for _ in range(max_length):
                cx += dx
                cy += dy
                if not (0 <= cx < LEVEL_WIDTH and 0 <= cy < LEVEL_HEIGHT):
                    return None
                if self.grid[cy][cx] == FLOOR:
                    return cx, cy
            return None

        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        random.shuffle(directions)  # Randomize digging order

        tunnels_dug = 0
        for dx, dy in directions:
            target = find_nearest_open(x, y, dx, dy)
            if target:
                cx, cy = x, y
                while (cx, cy) != target:
                    self.grid[cy][cx] = FLOOR
                    cx += dx
                    cy += dy
                tunnels_dug += 1
            
            if tunnels_dug >= 2:  # Ensure at least two tunnels are dug
                break

        # If no tunnels were dug, create a room around the player
        if tunnels_dug == 0:
            room_size = 5
            for ry in range(max(0, y - room_size), min(LEVEL_HEIGHT, y + room_size + 1)):
                for rx in range(max(0, x - room_size), min(LEVEL_WIDTH, x + room_size + 1)):
                    self.grid[ry][rx] = FLOOR

        # Ensure the starting point is also a floor
        self.grid[y][x] = FLOOR



    def handle_trap_fall(self, x, y):
        self.grid[y][x] = FLOOR  # Ensure the landing spot is a floor
        self.dig_tunnels(x, y)   # Dig tunnels from the landing spot
        # sound_manager.set_volume('trapdoor', 0.1)  # Adjust as needed
        sound_manager.play('trapdoor')
        # return x, y  # Return the landing coordinates
        # This method should return the coordinates for the player after falling
        return self.trapdoor_entry_position


class Game:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.FULLSCREEN)
        pygame.display.set_caption("YARL - Yet Another RogueLike v1.2")
        self.clock = pygame.time.Clock()
        self.current_level = 0
        self.levels = {}
        self.font = pygame.font.Font(None, 36)
        self.last_move_time = 0
        self.move_direction = None
        self.fog_of_war = [[False for _ in range(LEVEL_WIDTH)] for _ in range(LEVEL_HEIGHT)]
        self.visibility_radius = 20
        self.mouse_held = False
        self.last_mouse_move_time = 0
        self.sound_enabled = True
        self.attack_key_held = False
        self.last_attack_time = 0
        self.sword_level = 1
        self.monsters_slain = 0
        self.sentinel_monsters_slain = 0
        self.hp_healed_with_ep = 0

        self.inventory = Inventory()
        self.inventory.add_item(Sword(1))  # Start with a level 1 sword
        self.active_item = Sword(1)  # Start with a level 1 sword
        self.pickaxe = None
        self.rocks = 0
        self.rock_collection_progress = 0

        self.pass_key_held = False
        self.pass_key_first_press_time = 0
        self.last_pass_time = 0
        self.pass_repeat_delay = 300  # 0.3 seconds in milliseconds
        self.pass_repeat_interval = 50  # 0.05 seconds (20 times per second) in milliseconds

        self.path_to_clicked_location = []
        self.last_path_move_time = 0

        self.right_mouse_held = False
        self.last_right_mouse_attack_time = 0

        self.camera_x = 0
        self.camera_y = 0

        self.start_time = pygame.time.get_ticks()
        self.time_played = 0
        self.save_file_path = os.path.join(os.path.dirname(__file__), 'savegame.json')
        
        if os.path.exists(self.save_file_path):
            self.show_load_option()
        else:
            self.new_game()


    def new_game(self):
        self.current_level = 0
        self.levels = {}
        initial_level = self.get_or_create_level(self.current_level)
        spawn_x, spawn_y = initial_level.find_spawn_location()
        self.player = GameObject(spawn_x, spawn_y, DARK_BLUE, 'O')
        self.direction = "North"  # Set initial direction
        self.player_hp = 5000
        self.player_energy = 0
        self.last_move_time = 0
        self.move_direction = None
        self.weapon = Weapon("Level 1 Sword", 10)
        self.fog_of_war = [[False for _ in range(LEVEL_WIDTH)] for _ in range(LEVEL_HEIGHT)]
        self.visibility_radius = 20
        self.monsters_slain = 0
        self.sentinel_monsters_slain = 0
        self.hits_taken = 0
        self.steps_walked = 0
        self.traps_fallen = 0
        self.levels_cleared = 0        
        self.start_time = pygame.time.get_ticks()
        self.time_played = 0
        self.deepest_level_visited = 0
        self.sword_level = 1
        self.active_item = Sword(self.sword_level)
        self.pickaxe = None
        self.rocks = 0

        self.weapon = Weapon(f"Level {self.sword_level} Sword", 10)



    def play_sound(self, sound):
        sound_manager.play(sound)


    def toggle_sound(self):
        self.sound_enabled = sound_manager.toggle()
        # print("Sound:", "Enabled" if self.sound_enabled else "Disabled")



    def check_level_cleared(self):
        current_level = self.get_or_create_level(self.current_level)
        if not current_level.monsters and not current_level.cleared:
            self.levels_cleared += 1
            current_level.cleared = True


    def save_game(self):
        save_data = {
            'player_x': self.player.x,
            'player_y': self.player.y,
            'direction': self.direction,
            'player_hp': self.player_hp,
            'hp_healed': self.hp_healed_with_ep,
            'player_energy': self.player_energy,
            'current_level': self.current_level,
            'monsters_slain': self.monsters_slain,
            'sentinel_monsters_slain': self.sentinel_monsters_slain,
            'hits_taken': self.hits_taken,
            'steps_walked': self.steps_walked,
            'traps_fallen': self.traps_fallen,
            'levels_cleared': self.levels_cleared,
            'time_played': self.time_played,
            'deepest_level_visited': self.deepest_level_visited,
            'sound_enabled': self.sound_enabled,
            'sword_level': self.sword_level,
            'pickaxe_durability': self.pickaxe.durability if self.pickaxe else None,
            'rocks': self.rocks,
            'active_item_type': type(self.active_item).__name__,
            # 'has_mined': self.levels[self.deepest_level_visited].has_mined if self.deepest_level_visited in self.levels else False,
            'levels': {}
        }
        """
        for level_num, level in self.levels.items():
            is_deepest_level = (level_num == self.deepest_level_visited)
            save_data['levels'][level_num] = level.to_dict(is_deepest_level)
            save_data['has_mined'] = level.has_mined  # Include the has_mined flag in the save data
        """
        for level_num, level in self.levels.items():
            is_deepest_level = (level_num == self.deepest_level_visited)
            level_data = level.to_dict(is_deepest_level)
            level_data['has_mined'] = level.has_mined  # Correctly place has_mined inside the level data
            save_data['levels'][level_num] = level_data


        compressed_data = lz4.block.compress(json.dumps(save_data).encode('utf-8'))
        with open(self.save_file_path, 'wb') as f:
            f.write(compressed_data)



    def load_game(self):
        try:
            with open(self.save_file_path, 'rb') as f:
                compressed_data = f.read()
            save_data = json.loads(lz4.block.decompress(compressed_data).decode('utf-8'))
            
            # Create player if it doesn't exist
            if not hasattr(self, 'player'):
                self.player = GameObject(save_data['player_x'], save_data['player_y'], DARK_BLUE, 'O')
            else:
                self.player.x = save_data['player_x']
                self.player.y = save_data['player_y']

            self.direction = save_data['direction']
            self.player_hp = save_data['player_hp']
            # self.hp_healed_with_ep = save_data.get['hp_healed']           
            self.hp_healed_with_ep = save_data.get('hp_healed', 0)
            self.player_energy = save_data['player_energy']
            self.current_level = save_data['current_level']
            self.monsters_slain = save_data['monsters_slain']
            self.sentinel_monsters_slain = save_data['sentinel_monsters_slain']
            self.hits_taken = save_data['hits_taken']
            self.steps_walked = save_data['steps_walked']
            self.traps_fallen = save_data['traps_fallen']
            self.levels_cleared = save_data['levels_cleared']
            self.time_played = save_data['time_played']
            self.deepest_level_visited = save_data['deepest_level_visited']
            self.sound_enabled = save_data['sound_enabled']
            sound_manager.enabled = self.sound_enabled

            # Load inventory
            self.sword_level = save_data['sword_level']
            self.pickaxe = Pickaxe(save_data['pickaxe_durability']) if save_data['pickaxe_durability'] is not None else None
            self.rocks = save_data['rocks']

            if save_data['active_item_type'] == 'Sword':
                self.active_item = Sword(self.sword_level)
            elif save_data['active_item_type'] == 'Pickaxe':
                self.active_item = self.pickaxe
            else:
                self.active_item = Rocks(self.rocks)

            self.levels = {}
            """
            for level_num, level_data in save_data['levels'].items():
                is_deepest_level = (int(level_num) == self.deepest_level_visited)
                # Load the has_mined flag for each level
                level.has_mined = level_data.get('has_mined', False)

                self.levels[int(level_num)] = Level.from_dict(level_data, is_deepest_level)
                if is_deepest_level:
                    level.has_mined = save_data.get('has_mined', False)
            """
            for level_num, level_data in save_data['levels'].items():
                is_deepest_level = (int(level_num) == self.deepest_level_visited)
                level = Level.from_dict(level_data, is_deepest_level)

                # Load the has_mined flag for each level
                level.has_mined = level_data.get('has_mined', False)

                self.levels[int(level_num)] = level


            
            # Update the current level's fog of war
            current_level = self.get_or_create_level(self.current_level)
            self.fog_of_war = current_level.fog_of_war
            
            return True
        except FileNotFoundError:
            return False




    def show_load_option(self):
        loading = True
        while loading:
            self.screen.fill(BLACK)
            text1 = self.font.render("Savegame found. Do you want to load it?", True, WHITE)
            text2 = self.font.render("Press L to load, N for new game", True, WHITE)
            self.screen.blit(text1, (SCREEN_WIDTH // 2 - text1.get_width() // 2, SCREEN_HEIGHT // 2 - 50))
            self.screen.blit(text2, (SCREEN_WIDTH // 2 - text2.get_width() // 2, SCREEN_HEIGHT // 2 + 50))
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_l:
                        self.load_game()
                        loading = False
                    elif event.key == pygame.K_n:
                        self.new_game()
                        loading = False
                elif event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()


    def get_or_create_level(self, level_number, entry_type='stairs'):
        if level_number not in self.levels:
            self.levels[level_number] = Level(level_number, entry_type=entry_type)
        return self.levels[level_number]

    #----------------------------------------------------------------------------------------

    """
    def is_healing_allowed(self):
        current_level = self.get_or_create_level(self.current_level)
        return (
            self.current_level == self.deepest_level_visited and
            len(current_level.monsters) == current_level.initial_monster_count and
            current_level.placed_rocks_count == 0
        )
    """

    def is_healing_allowed(self):
        current_level = self.get_or_create_level(self.current_level)
        # Calculate the threshold for 5% of the original monster count
        threshold_monster_count = current_level.initial_monster_count * 0.05
        
        return (
            self.current_level == self.deepest_level_visited and
            len(current_level.monsters) >= threshold_monster_count and  # At least 1/20th of the initial monsters are present
            current_level.placed_rocks_count == 0  and
            not current_level.has_mined  # Use has_mined from the current level
        )



    def upgrade_sword(self):
        self.sword_level += 1
        new_damage = 10 + (self.sword_level - 1) * 2
        self.weapon = Weapon(f"Level {self.sword_level} Sword", new_damage)
        # print(f"Upgraded to {self.weapon.name} (DMG: {self.weapon.damage})")



    def use_active_item(self):
        # print(f"Using active item: {type(self.active_item).__name__}")  # Debug print
        if isinstance(self.active_item, Sword):
            self.attack()
        elif isinstance(self.active_item, Pickaxe):
            self.use_pickaxe()
        elif isinstance(self.active_item, Rocks):
            self.place_rocks()
        #else:
        #    print(f"Unknown active item: {type(self.active_item).__name__}")  # Debug print



    def use_pickaxe(self):
        if not isinstance(self.active_item, Pickaxe) or self.active_item.durability <= 0:
            return

        dx, dy = self.get_direction_vector()
        target_x, target_y = self.player.x + dx, self.player.y + dy
        current_level = self.get_or_create_level(self.current_level)
        
        if current_level.grid[target_y][target_x] in [WALL, PLACED_ROCK]:
            self.rock_collection_progress += 1
            required_progress = 4 if current_level.grid[target_y][target_x] == WALL else 2
            
            if self.rock_collection_progress >= required_progress:
                if current_level.grid[target_y][target_x] == WALL:
                    self.active_item.durability -= 1
                    self.has_mined = True
                elif current_level.grid[target_y][target_x] == PLACED_ROCK:
                    # self.placed_rocks_count -= 1
                    current_level.placed_rocks_count -= 1

                current_level.grid[target_y][target_x] = FLOOR
                self.rocks += 1
                self.rock_collection_progress = 0
                sound_manager.play('collect_rock')
            else:
                sound_manager.play('pickaxe_hit')  # You might want to add this sound
        
        self.handle_monster_turn()


    def place_rocks(self):
        if not isinstance(self.active_item, Rocks) or self.rocks <= 0:
            return

        dx, dy = self.get_direction_vector()
        place_x, place_y = self.player.x + dx, self.player.y + dy
        current_level = self.get_or_create_level(self.current_level)
        
        if current_level.grid[place_y][place_x] == FLOOR:
            current_level.grid[place_y][place_x] = PLACED_ROCK
            self.rocks -= 1
            self.placed_rocks_count += 1
            sound_manager.play('place_rock')  # You'll need to add this sound
        
        self.handle_monster_turn()


    def switch_item(self):
        if isinstance(self.active_item, Sword):
            self.active_item = self.pickaxe if self.pickaxe else (Rocks(self.rocks) if self.rocks > 0 else Sword(self.sword_level))
        elif isinstance(self.active_item, Pickaxe):
            self.active_item = Rocks(self.rocks) if self.rocks > 0 else Sword(self.sword_level)
        else:  # Rocks
            self.active_item = Sword(self.sword_level)
        # print(f"Switched to: {type(self.active_item).__name__}")  # Debug print
        self.handle_monster_turn()  # Switching items counts as a turn


    def get_direction_vector(self):
        if self.direction == "North":
            return 0, -1
        elif self.direction == "South":
            return 0, 1
        elif self.direction == "East":
            return 1, 0
        elif self.direction == "West":
            return -1, 0


    def collect_rocks(self):
        if isinstance(self.inventory.active_item, Pickaxe):
            if self.inventory.active_item.durability > 0:
                dx, dy = self.get_direction_vector()
                wall_x, wall_y = self.player.x + dx, self.player.y + dy
                current_level = self.get_or_create_level(self.current_level)
                if current_level.grid[wall_y][wall_x] == WALL:
                    self.inventory.active_item.durability -= 1
                    current_level.grid[wall_y][wall_x] = FLOOR
                    rocks = self.inventory.get_rocks()
                    if rocks:
                        rocks.amount += 1
                    else:
                        self.inventory.add_item(Rocks(1))


    def place_rocks(self):
        # print(f"Attempting to place rocks. Rocks: {self.rocks}")  # Debug print
        if not isinstance(self.active_item, Rocks) or self.rocks <= 0:
            # print("Cannot place rocks: invalid active item or no rocks left")  # Debug print
            return

        dx, dy = self.get_direction_vector()
        place_x, place_y = self.player.x + dx, self.player.y + dy
        current_level = self.get_or_create_level(self.current_level)
        
        # print(f"Trying to place rock at: ({place_x}, {place_y})")  # Debug print
        # print(f"Tile at target location: {current_level.grid[place_y][place_x]}")  # Debug print
        
        if current_level.grid[place_y][place_x] == FLOOR:
            current_level.grid[place_y][place_x] = PLACED_ROCK
            self.rocks -= 1
            # print(f"Rock placed. Remaining rocks: {self.rocks}")  # Debug print
            sound_manager.play('place_rock')
        #else:
        #    print("Cannot place rock: target location is not an empty floor")  # Debug print
        
        self.handle_monster_turn()




    def perform_attack(self):
        current_time = pygame.time.get_ticks()
        if current_time - self.last_attack_time >= MOVE_REPEAT_INTERVAL:
            self.last_attack_time = current_time
            self.attack()


    def handle_mouse_movement(self, mouse_pos):
        start_x, start_y = self.get_camera_offset()
        tile_x = start_x + (mouse_pos[0] - 40) // TILE_SIZE
        tile_y = start_y + (mouse_pos[1] - 40) // TILE_SIZE

        dx = tile_x - self.player.x
        dy = tile_y - self.player.y

        if abs(dx) > abs(dy):
            self.move_player(1 if dx > 0 else -1, 0)
        elif dy != 0:
            self.move_player(0, 1 if dy > 0 else -1)



    def pass_turn(self):
        if self.is_healing_allowed():
            if self.player_energy > 0 and self.player_hp < 5000:
                old_hp = self.player_hp
                self.player_energy -= 1
                self.player_hp = min(5000, self.player_hp + 1)
                self.hp_healed_with_ep += self.player_hp - old_hp
                self.play_sound('tiny_heal_tick')
        
        # Handle monster movements and attacks
        self.handle_monster_turn()
        
        # Update game state
        self.time_played += 1
        self.update_fog_of_war()
        self.check_level_cleared()



    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                self.save_game()
                return False
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT):
                    self.move_direction = event.key
                    self.last_move_time = pygame.time.get_ticks()
                    # Immediate move on key press
                    if event.key == pygame.K_UP:
                        self.move_player(0, -1)
                    elif event.key == pygame.K_DOWN:
                        self.move_player(0, 1)
                    elif event.key == pygame.K_LEFT:
                        self.move_player(-1, 0)
                    elif event.key == pygame.K_RIGHT:
                        self.move_player(1, 0)

                elif event.key == pygame.K_u:
                    self.use_stairs('up')
                elif event.key == pygame.K_d:
                    self.use_stairs('down')
                elif event.key == pygame.K_a:
                    self.attack_key_held = True
                    self.perform_attack()
                elif event.key == pygame.K_i:
                    self.switch_item()

                if event.key == pygame.K_c:
                    self.center_view()
                elif event.key == pygame.K_s:
                    self.toggle_sound()
                elif event.key == pygame.K_PLUS:  # or event.key == pygame.K_EQUALS:  # Both + and = keys (since + is Shift+=)
                    new_volume = sound_manager.adjust_volume(0.05)
                    # print(f"Volume increased to {new_volume:.2f}")
                elif event.key == pygame.K_MINUS:
                    new_volume = sound_manager.adjust_volume(-0.05)
                    # print(f"Volume decreased to {new_volume:.2f}")

                elif event.key == pygame.K_p:
                    if not self.pass_key_held:
                        self.pass_key_held = True
                        self.pass_key_first_press_time = pygame.time.get_ticks()
                        self.last_pass_time = self.pass_key_first_press_time
                        self.pass_turn()

            if event.type == pygame.KEYUP:
                if event.key == self.move_direction:
                    self.move_direction = None
                elif event.key == pygame.K_a:
                    self.attack_key_held = False
                elif event.key == pygame.K_p:
                    self.pass_key_held = False

            if event.type == pygame.MOUSEWHEEL:
                if event.y < 0:  # Scrolling down
                    self.cycle_item_forward()
                elif event.y > 0:  # Scrolling up
                    self.cycle_item_backward()

            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button
                    """
                    self.handle_mouse_movement(event.pos)
                    self.mouse_held = True
                    self.last_mouse_move_time = pygame.time.get_ticks()
                    """

                    mouse_pos = event.pos
                    clicked_x, clicked_y = self.screen_to_world_coords(mouse_pos[0], mouse_pos[1])
                    self.move_to_clicked_location(clicked_x, clicked_y)

                elif event.button == 2:  # Middle click
                    self.center_view()

                elif event.button == 3:  # Right mouse button
                    self.right_mouse_held = True
                    self.use_active_item()


            if event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:  # Left mouse button                    
                    self.mouse_held = False
                    self.path_to_clicked_location = []  # Clear the path when mouse is released
                elif event.button == 3:  # Right mouse button
                    self.right_mouse_held = False
        return True


    def screen_to_world_coords(self, screen_x, screen_y):
        # Adjust for the HUD height
        screen_y -= 40  

        # Get the current camera offset
        camera_x, camera_y = self.get_camera_offset()

        # Calculate the world coordinates
        world_x = camera_x + (screen_x // TILE_SIZE)
        world_y = camera_y + (screen_y // TILE_SIZE)

        return world_x, world_y



    def cycle_item_forward(self):
        if isinstance(self.active_item, Sword):
            self.active_item = self.pickaxe if self.pickaxe else (Rocks(self.rocks) if self.rocks > 0 else Sword(self.sword_level))
        elif isinstance(self.active_item, Pickaxe):
            self.active_item = Rocks(self.rocks) if self.rocks > 0 else Sword(self.sword_level)
        else:  # Rocks
            self.active_item = Sword(self.sword_level)
        self.handle_monster_turn()  # Switching items counts as a turn


    def cycle_item_backward(self):
        if isinstance(self.active_item, Sword):
            self.active_item = Rocks(self.rocks) if self.rocks > 0 else (self.pickaxe if self.pickaxe else Sword(self.sword_level))
        elif isinstance(self.active_item, Pickaxe):
            self.active_item = Sword(self.sword_level)
        else:  # Rocks
            self.active_item = self.pickaxe if self.pickaxe else Sword(self.sword_level)
        self.handle_monster_turn()  # Switching items counts as a turn



    def use_stairs(self, direction):
        current_level = self.get_or_create_level(self.current_level)
        player_tile = current_level.grid[self.player.y][self.player.x]
        
        if direction == 'up' and player_tile == STAIRS_UP:
            self.current_level -= 1
            next_level = self.get_or_create_level(self.current_level)
            self.player.x, self.player.y = next_level.stairs_down_position
        elif direction == 'down' and player_tile == STAIRS_DOWN:
            self.current_level += 1
            if self.current_level > self.deepest_level_visited:
                self.deepest_level_visited = self.current_level
                sound_manager.play('new_level')                
            next_level = self.get_or_create_level(self.current_level)
            self.player.x, self.player.y = next_level.stairs_up_position
        
        self.update_fog_of_war()



    def distance_to_nearest_monster(self):
        current_level = self.get_or_create_level(self.current_level)
        if not current_level.monsters:
            return float('inf')
        
        distances = [max(abs(self.player.x - m.x), abs(self.player.y - m.y)) for m in current_level.monsters]
        return min(distances)

    #-----------------------------------------------------------------------------------------------------------------------------------------------


    def move_along_path(self):
        if not self.path_to_clicked_location:
            # print("Path is empty, stopping movement")  # Debug print
            return

        next_step = self.path_to_clicked_location[0]
        dx = next_step[0] - self.player.x
        dy = next_step[1] - self.player.y
        
        # Ensure orthogonal movement
        if dx != 0:
            dx = 1 if dx > 0 else -1
            dy = 0
        elif dy != 0:
            dy = 1 if dy > 0 else -1
            dx = 0
        
        current_level = self.get_or_create_level(self.current_level)
        new_x, new_y = self.player.x + dx, self.player.y + dy

        if current_level.is_valid_move(new_x, new_y):
            # Move the player regardless of whether it's a trap or not
            self.move_player(dx, dy)
            
            # Check for trap after moving
            if current_level.grid[new_y][new_x] == TRAP:
                # print("Encountered a trap")  # Debug print
                self.handle_trap(new_x, new_y, current_level)
                self.path_to_clicked_location = []  # Clear the path after falling
            else:
                if self.path_to_clicked_location:  # Check if path is not empty before popping
                    self.path_to_clicked_location.pop(0)
            
            # print(f"Moved to {self.player.x}, {self.player.y}")  # Debug print
        else:
            # print(f"Move blocked at {new_x}, {new_y}")  # Debug print
            # If the move was blocked, clear the path
            self.path_to_clicked_location = []

        #if not self.path_to_clicked_location:
        #    print("Reached end of path or path blocked")  # Debug print




    def move_to_clicked_location(self, target_x, target_y):
        current_level = self.get_or_create_level(self.current_level)
        start = (self.player.x, self.player.y)
        goal = (target_x, target_y)
        
        self.clicked_location = (target_x, target_y)  # Store the clicked location

        # print(f"Attempting to move from {start} to {goal}")  # Debug print
        # print(f"Target tile type: {current_level.grid[target_y][target_x]}")  # Debug print
        
        # Remove the floor check and always attempt pathfinding
        path = current_level.player_astar(start, goal)
        if path:
            # print(f"Path found: {path}")  # Debug print
            self.path_to_clicked_location = path[1:]  # Remove the starting position
        #else:
        #    print("No path found")  # Debug print



    def move_player(self, dx, dy):
        new_x, new_y = self.player.x + dx, self.player.y + dy
        current_level = self.get_or_create_level(self.current_level)
        if dx == 1:
            self.direction = "East"
        elif dx == -1:
            self.direction = "West"
        elif dy == 1:
            self.direction = "South"
        elif dy == -1:
            self.direction = "North"
        if 0 <= new_x < LEVEL_WIDTH and 0 <= new_y < LEVEL_HEIGHT:
            if current_level.grid[new_y][new_x] not in [WALL, PLACED_ROCK]:
                if current_level.grid[new_y][new_x] == CHEST and not current_level.chest_opened:
                    self.open_chest(current_level)
                else:
                    monster = next((m for m in current_level.monsters if m.x == new_x and m.y == new_y), None)
                    if monster:
                        if isinstance(self.active_item, Sword):
                            self.attack_monster(monster)
                        else:
                            # Player tried to move into a monster without a sword
                            return
                    else:
                        sound_manager.play('move')
                        self.player.x, self.player.y = new_x, new_y
                        self.steps_walked += 1

                        # Energy-based healing
                        if self.is_healing_allowed():
                            if self.player_energy > 0 and self.player_hp < 5000:
                                old_hp = self.player_hp
                                self.player_energy -= 1
                                self.player_hp = min(5000, self.player_hp + 1)
                                self.hp_healed_with_ep += self.player_hp - old_hp
                                sound_manager.play('heal_tick')


                    if current_level.grid[new_y][new_x] == TRAP:
                        self.handle_trap(new_x, new_y, current_level)
                        return True


                self.handle_monster_turn()  # Call this after player's action
                return True

        self.update_fog_of_war()
        return False



    def handle_trap(self, x, y, current_level):
        # Reveal the trap if it wasn't already
        if (x, y) not in current_level.revealed_traps:
            current_level.reveal_trap(x, y)
            # Apply full trap effect for hidden traps
            self.player_hp = int(self.player_hp * 0.85)
            self.player_energy = int(self.player_energy * 0.7)
        else:
            # Known trap effect
            energy_cost = random.randint(500, 1000)
            hp_reduction = max(1, int(self.player_hp * 0.01))  # 1% HP reduction, minimum 1
            
            self.player_hp -= hp_reduction
            if self.player_energy >= energy_cost:
                self.player_energy -= energy_cost
            else:
                additional_hp_loss = energy_cost - self.player_energy
                self.player_hp -= additional_hp_loss
                self.player_energy = 0

        self.hits_taken += 1
        self.traps_fallen += 1
        
        sound_manager.play('trapdoor')
        
        self.current_level += 1
        self.deepest_level_visited = max(self.deepest_level_visited, self.current_level)
        next_level = self.get_or_create_level(self.current_level, entry_type='trapdoor')
        
        self.player.x, self.player.y = next_level.find_spawn_location()
        self.update_fog_of_war()
        return True  # Trap was triggered and handled




    def play_chest_open_sound(self):
        sound_manager.play('chest_open')


    def open_chest(self, level):
        roll = random.random()
        if roll < 0.2:
            self.sword_level += 1
            self.active_item = Sword(self.sword_level)
        elif roll < 0.95:
            if self.pickaxe:
                self.pickaxe.durability += 20
            else:
                self.pickaxe = Pickaxe(20)
            if isinstance(self.active_item, Pickaxe):
                self.active_item = self.pickaxe
        else:
            self.rocks += 20
            if isinstance(self.active_item, Rocks):
                self.active_item = Rocks(self.rocks)
        
        level.chest_opened = True
        level.grid[level.chest_position[1]][level.chest_position[0]] = FLOOR
        if not level.first_chest_found:
            level.first_chest_found = True
        self.play_chest_open_sound()


    #---------------------------------------------------------------------------------------

    def attack_monster(self, monster):
        monster.current_hp -= self.active_item.damage
        self.play_sound('attack')
        if monster.current_hp <= 0:
            current_level = self.get_or_create_level(self.current_level)
            current_level.monsters.remove(monster)
            self.monsters_slain += 1
            if monster.monster_type == "sentinel":
                self.sentinel_monsters_slain += 1
            self.play_sound('monster_death')
            
            # Heal player and add energy
            if monster.monster_type == "regular":
                heal_amount = (monster.max_hp * 11) // 4
                energy_bonus = monster.max_hp
            else:  # sentinel
                heal_amount = (monster.max_hp * 11) // 4
                energy_bonus = monster.max_hp * 2
            
            old_hp = self.player_hp
            self.player_hp = min(5000, self.player_hp + heal_amount)
            self.hp_healed_with_ep += self.player_hp - old_hp
            self.player_energy = min(25000, self.player_energy + energy_bonus)
            
            # Check if all monsters are cleared
            if not current_level.monsters:
                old_hp = self.player_hp
                bonus_heal = (monster.max_hp * 5)
                self.player_hp = min(5000, self.player_hp + bonus_heal)
                self.hp_healed_with_ep += self.player_hp - old_hp
        
        self.handle_monster_turn()




    def handle_monster_turn(self):
        current_level = self.get_or_create_level(self.current_level)
        attacking_monsters = current_level.move_monsters(self.player.x, self.player.y)
        for monster in attacking_monsters:
            self.player_hp -= monster.damage
            self.player_hp = max(0, self.player_hp)
            self.hits_taken += 1
            self.play_sound('player_hit')



    def attack(self):
        if not isinstance(self.active_item, Sword):
            return  # Can only attack with a sword

        current_level = self.get_or_create_level(self.current_level)
        adjacent_monsters = []
        
        for dx, dy in [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]:
            attack_x, attack_y = self.player.x + dx, self.player.y + dy
            for monster in current_level.monsters:
                if monster.x == attack_x and monster.y == attack_y:
                    adjacent_monsters.append(monster)
        
        if adjacent_monsters:
            target_monster = min(adjacent_monsters, key=lambda m: m.current_hp)
            self.attack_monster(target_monster)
        
        self.handle_monster_turn()



    def update(self):
        current_time = pygame.time.get_ticks()
        if self.move_direction:
            if current_time - self.last_move_time >= MOVE_REPEAT_INTERVAL:
                self.last_move_time = current_time
                if self.move_direction == pygame.K_UP:
                    self.move_player(0, -1)
                elif self.move_direction == pygame.K_DOWN:
                    self.move_player(0, 1)
                elif self.move_direction == pygame.K_LEFT:
                    self.move_player(-1, 0)
                elif self.move_direction == pygame.K_RIGHT:
                    self.move_player(1, 0)


        elif self.path_to_clicked_location:
            if current_time - self.last_path_move_time >= MOVE_REPEAT_INTERVAL:
                self.last_path_move_time = current_time
                self.move_along_path()

        elif self.mouse_held:
            if current_time - self.last_mouse_move_time >= MOVE_REPEAT_INTERVAL:
                self.last_mouse_move_time = current_time
                self.handle_mouse_movement(pygame.mouse.get_pos())


        # Handle pass turn key repeat
        if self.pass_key_held:
            time_since_first_press = current_time - self.pass_key_first_press_time
            if time_since_first_press >= self.pass_repeat_delay:
                time_since_last_pass = current_time - self.last_pass_time
                if time_since_last_pass >= self.pass_repeat_interval:
                    self.pass_turn()
                    self.last_pass_time = current_time

        # Handle attack key and right mouse button repeat
        if self.attack_key_held or self.right_mouse_held:
            if current_time - self.last_attack_time >= MOVE_REPEAT_INTERVAL:
                self.last_attack_time = current_time
                self.use_active_item()
        
        self.time_played = (pygame.time.get_ticks() - self.start_time) // 1000



    def update_fog_of_war(self):
        current_level = self.get_or_create_level(self.current_level)
        for y in range(LEVEL_HEIGHT):
            for x in range(LEVEL_WIDTH):
                distance = max(abs(self.player.x - x), abs(self.player.y - y))
                if distance <= self.visibility_radius:
                    current_level.fog_of_war[y][x] = True
        self.fog_of_war = current_level.fog_of_war

    def render(self):
        self.screen.fill(BLACK)
        current_level = self.get_or_create_level(self.current_level)
        
        # Calculate the visible area
        visible_width = SCREEN_WIDTH // TILE_SIZE
        visible_height = (SCREEN_HEIGHT - 40) // TILE_SIZE

        # start_x = max(0, min(self.player.x - visible_width // 2, LEVEL_WIDTH - visible_width))
        # start_y = max(0, min(self.player.y - visible_height // 2, LEVEL_HEIGHT - visible_height))
        # Get the camera offset
        start_x, start_y = self.get_camera_offset()

        """
        # for debugging ------------------------------------
        # Render clicked location marker if exists
        if hasattr(self, 'clicked_location'):
            clicked_x, clicked_y = self.clicked_location
            screen_x = (clicked_x - start_x) * TILE_SIZE
            screen_y = (clicked_y - start_y) * TILE_SIZE + 40
            pygame.draw.rect(self.screen, (255, 255, 0), (screen_x, screen_y, TILE_SIZE, TILE_SIZE), 2)
        #----------------------------------------------------
        """
        for y in range(visible_height):
            for x in range(visible_width):
                world_x = start_x + x
                world_y = start_y + y

                if 0 <= world_x < LEVEL_WIDTH and 0 <= world_y < LEVEL_HEIGHT:
                    if self.fog_of_war[world_y][world_x]:

                        tile = current_level.grid[world_y][world_x]

                        if tile == WALL:
                            color = BLACK
                        elif tile == PLACED_ROCK:
                            color = BLUISH_GREY
                        else:
                            color = DARK_GREY
                        pygame.draw.rect(self.screen, color, (x * TILE_SIZE, y * TILE_SIZE + 40, TILE_SIZE, TILE_SIZE))

                        
                        if tile == STAIRS_DOWN:
                            pygame.draw.line(self.screen, GREY, (x * TILE_SIZE, y * TILE_SIZE + 40), 
                                             ((x + 1) * TILE_SIZE, (y + 1) * TILE_SIZE + 40), 2)
                        elif tile == STAIRS_UP:
                            pygame.draw.line(self.screen, GREY, (x * TILE_SIZE, (y + 1) * TILE_SIZE + 40), 
                                             ((x + 1) * TILE_SIZE, y * TILE_SIZE + 40), 2)
                        elif (world_x, world_y) in current_level.revealed_traps:
                            pygame.draw.line(self.screen, RED, (x * TILE_SIZE, y * TILE_SIZE + 40), 
                                             ((x + 1) * TILE_SIZE, (y + 1) * TILE_SIZE + 40), 2)
                            pygame.draw.line(self.screen, RED, (x * TILE_SIZE, (y + 1) * TILE_SIZE + 40), 
                                             ((x + 1) * TILE_SIZE, y * TILE_SIZE + 40), 2)

                        elif tile == CHEST and not current_level.chest_opened:
                            pygame.draw.rect(self.screen, (139, 69, 19), (x * TILE_SIZE, y * TILE_SIZE + 40, TILE_SIZE, TILE_SIZE * 0.8))
                            pygame.draw.rect(self.screen, YELLOW, (x * TILE_SIZE, y * TILE_SIZE + 40, TILE_SIZE, TILE_SIZE * 0.8), 2)
                            for i in range(3):
                                pygame.draw.circle(self.screen, GREY, (x * TILE_SIZE + (i+1) * TILE_SIZE // 4, y * TILE_SIZE + 45), 2)


                    else:
                        # Draw unexplored areas
                        pygame.draw.rect(self.screen, (20, 20, 20), (x * TILE_SIZE, y * TILE_SIZE + 40, TILE_SIZE, TILE_SIZE))


        # Render both monster types
        for monster in current_level.monsters:
            if (start_x <= monster.x < start_x + visible_width and 
                start_y <= monster.y < start_y + visible_height and
                self.fog_of_war[monster.y][monster.x]):
                screen_x = (monster.x - start_x) * TILE_SIZE
                screen_y = (monster.y - start_y) * TILE_SIZE + 40
                center_x = screen_x + TILE_SIZE // 2
                center_y = screen_y + TILE_SIZE // 2
                
                if monster.monster_type == "regular":
                    size = int(TILE_SIZE * monster.size_factor)
                    pygame.draw.polygon(self.screen, GREEN, 
                                        [(center_x, center_y - size // 2),
                                         (center_x - size // 2, center_y + size // 2),
                                         (center_x + size // 2, center_y + size // 2)])

                elif monster.monster_type == "sentinel":
                    size_factor = monster.size_factor
                    width = int(TILE_SIZE * 0.5 * size_factor)  # Half as wide as a tile
                    height = int(TILE_SIZE * size_factor)  # As long as a tile
                    
                    # Determine the position and orientation
                    if monster.direction in ["left", "right"]:
                        rect = pygame.Rect(center_x - height // 2, center_y - width // 2, height, width)
                    else:  # "up" or "down"
                        rect = pygame.Rect(center_x - width // 2, center_y - height // 2, width, height)
                    
                    # Draw main body
                    pygame.draw.rect(self.screen, BRIGHT_ORANGE, rect)
                    
                    # Draw tail (2 pixels)
                    tail_size = max(2, int(2 * size_factor))  # Ensure at least 2 pixels
                    if monster.direction == "right":
                        pygame.draw.rect(self.screen, BRIGHT_ORANGE, (rect.left, rect.centery - tail_size // 2, tail_size, tail_size))
                    elif monster.direction == "left":
                        pygame.draw.rect(self.screen, BRIGHT_ORANGE, (rect.right - tail_size, rect.centery - tail_size // 2, tail_size, tail_size))
                    elif monster.direction == "down":
                        pygame.draw.rect(self.screen, BRIGHT_ORANGE, (rect.centerx - tail_size // 2, rect.top, tail_size, tail_size))
                    else:  # up
                        pygame.draw.rect(self.screen, BRIGHT_ORANGE, (rect.centerx - tail_size // 2, rect.bottom - tail_size, tail_size, tail_size))
                    
                    # Draw eyes
                    eye_size = max(1, int(TILE_SIZE * 0.05 * size_factor))  # Ensure at least 1 pixel size
                    eye_offset = int(width * 0.2)
                    if monster.direction == "right":
                        pygame.draw.rect(self.screen, RED, (rect.right - eye_offset, rect.top + eye_offset, eye_size, eye_size))
                        pygame.draw.rect(self.screen, RED, (rect.right - eye_offset, rect.bottom - eye_offset - eye_size, eye_size, eye_size))
                    elif monster.direction == "left":
                        pygame.draw.rect(self.screen, RED, (rect.left + eye_offset - eye_size, rect.top + eye_offset, eye_size, eye_size))
                        pygame.draw.rect(self.screen, RED, (rect.left + eye_offset - eye_size, rect.bottom - eye_offset - eye_size, eye_size, eye_size))
                    elif monster.direction == "down":
                        pygame.draw.rect(self.screen, RED, (rect.left + eye_offset, rect.bottom - eye_offset - eye_size, eye_size, eye_size))
                        pygame.draw.rect(self.screen, RED, (rect.right - eye_offset - eye_size, rect.bottom - eye_offset - eye_size, eye_size, eye_size))
                    else:  # up
                        pygame.draw.rect(self.screen, RED, (rect.left + eye_offset, rect.top + eye_offset, eye_size, eye_size))
                        pygame.draw.rect(self.screen, RED, (rect.right - eye_offset - eye_size, rect.top + eye_offset, eye_size, eye_size))


        # Render player
        player_screen_x = (self.player.x - start_x) * TILE_SIZE
        player_screen_y = (self.player.y - start_y) * TILE_SIZE + 40
        pygame.draw.circle(self.screen, self.player.color, 
                           (player_screen_x + TILE_SIZE // 2, 
                            player_screen_y + TILE_SIZE // 2), TILE_SIZE // 2)

        # Add mini-map
        mini_map_size = 100
        mini_map_surface = pygame.Surface((mini_map_size, mini_map_size))
        mini_map_surface.fill((50, 50, 50))
        for y in range(LEVEL_HEIGHT):
            for x in range(LEVEL_WIDTH):
                if self.fog_of_war[y][x]:
                    color = (0, 0, 0) if current_level.grid[y][x] == WALL else (100, 100, 100) if current_level.grid[y][x] == PLACED_ROCK else (150, 150, 150)
                    pygame.draw.rect(mini_map_surface, color, (x * mini_map_size // LEVEL_WIDTH, y * mini_map_size // LEVEL_HEIGHT, 
                                     max(1, mini_map_size // LEVEL_WIDTH), max(1, mini_map_size // LEVEL_HEIGHT)))

        
        # Draw player position on mini-map
        pygame.draw.rect(mini_map_surface, (0, 0, 255), (self.player.x * mini_map_size // LEVEL_WIDTH, 
                         self.player.y * mini_map_size // LEVEL_HEIGHT, 2, 2))
        
        self.screen.blit(mini_map_surface, (SCREEN_WIDTH - mini_map_size - 10, SCREEN_HEIGHT - mini_map_size - 10))

        # Render the HUD
        monsters_left = len(current_level.monsters)

        volume_percentage = int(sound_manager.volume * 100)
        volume_text = f"Vol:{volume_percentage}%" if sound_manager.enabled else "Muted"

        hp_color = GREEN if self.player_hp > 2500 else YELLOW if self.player_hp > 1000 else RED
        energy_color = BLUE
        weapon_color = YELLOW
        
        hud_text = f"Lvl:{self.current_level + 1} | X:{self.player.x} Y:{self.player.y} | D:{self.direction} | "
        hud_surface = self.font.render(hud_text, True, WHITE)
        self.screen.blit(hud_surface, (10, 10))
        
        x_offset = hud_surface.get_width() + 10
        
        hp_text = f"HP:{self.player_hp}"
        hp_surface = self.font.render(hp_text, True, hp_color)
        self.screen.blit(hp_surface, (x_offset, 10))
        x_offset += hp_surface.get_width() + 10
        
        energy_text = f"EP:{self.player_energy}"
        energy_surface = self.font.render(energy_text, True, energy_color)
        self.screen.blit(energy_surface, (x_offset, 10))
        x_offset += energy_surface.get_width() + 10

        # Update HUD to show active item  ---------------------
        if isinstance(self.active_item, Sword):
            item_text = f"item:Sword[level:{self.active_item.level}]"
        elif isinstance(self.active_item, Pickaxe):
            item_text = f"item:Pickaxe[durability:{self.active_item.durability}]"
        else:  # Rocks
            item_text = f"item:Rocks[{self.rocks}]"
        #-------------------------------------------------------
        
        # weapon_text = f"W:{self.weapon.name}({self.weapon.damage})"
        weapon_text = item_text
        weapon_surface = self.font.render(weapon_text, True, weapon_color)
        self.screen.blit(weapon_surface, (x_offset, 10))
        x_offset += weapon_surface.get_width() + 10
        

        remaining_text = f"Mobs:{monsters_left} | {volume_text}"
        remaining_surface = self.font.render(remaining_text, True, WHITE)
        self.screen.blit(remaining_surface, (x_offset, 10))



        # Render HP bar
        pygame.draw.rect(self.screen, RED, (SCREEN_WIDTH - 210, 10, 200, 20))
        pygame.draw.rect(self.screen, (0, 255, 0), (SCREEN_WIDTH - 210, 10, min(200, self.player_hp // 25), 20))
        
        # Render Energy bar
        pygame.draw.rect(self.screen, BLUE, (SCREEN_WIDTH - 210, 35, 200, 10))
        pygame.draw.rect(self.screen, YELLOW, (SCREEN_WIDTH - 210, 35, min(200, self.player_energy // 125), 10))
        
        pygame.display.flip()


    def get_camera_offset(self):
        visible_width = SCREEN_WIDTH // TILE_SIZE
        visible_height = (SCREEN_HEIGHT - 40) // TILE_SIZE
        
        # Calculate the ideal center
        ideal_center_x = self.player.x - visible_width // 2
        ideal_center_y = self.player.y - visible_height // 2
        
        # Calculate the current view boundaries
        current_left = self.camera_x
        current_top = self.camera_y
        
        # Check if we need to scroll
        if abs(ideal_center_x - current_left) > 18 or abs(ideal_center_y - current_top) > 18:
            self.camera_x = max(0, min(ideal_center_x, LEVEL_WIDTH - visible_width))
            self.camera_y = max(0, min(ideal_center_y, LEVEL_HEIGHT - visible_height))
        
        return self.camera_x, self.camera_y


    def center_view(self):
        visible_width = SCREEN_WIDTH // TILE_SIZE
        visible_height = (SCREEN_HEIGHT - 40) // TILE_SIZE
        self.camera_x = max(0, min(self.player.x - visible_width // 2, LEVEL_WIDTH - visible_width))
        self.camera_y = max(0, min(self.player.y - visible_height // 2, LEVEL_HEIGHT - visible_height))


    def render_game_over(self):
        pygame.event.clear()

        sound_manager.play('death')
        # self.play_sound(death_sound)
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        overlay.set_alpha(180)
        overlay.fill((0, 0, 0))
        self.screen.blit(overlay, (0, 0))

        font = pygame.font.Font(None, 72)
        game_over_text = font.render("G A M E   O V E R", True, RED)
        subtitle_text = font.render("You have died, better luck next time.", True, WHITE)
        self.screen.blit(game_over_text, (SCREEN_WIDTH // 2 - game_over_text.get_width() // 2, SCREEN_HEIGHT // 2 - 150))
        self.screen.blit(subtitle_text, (SCREEN_WIDTH // 2 - subtitle_text.get_width() // 2, SCREEN_HEIGHT // 2 - 80))

        stats_font = pygame.font.Font(None, 36)

        stats = [
            f"Levels entered: {self.current_level + 1}",
            f"Regular monsters slain: {self.monsters_slain - self.sentinel_monsters_slain}",
            f"Sentinel monsters slain: {self.sentinel_monsters_slain}",
            f"Levels cleared: {self.levels_cleared}",
            f"Steps walked: {self.steps_walked}",
            f"Hits taken: {self.hits_taken}",
            f"Traps fallen: {self.traps_fallen}",
            f"HPs healed: {self.hp_healed_with_ep}",
            f"Energy remaining: {self.player_energy}",
            f"Sword level: {self.sword_level}",
            f"Pickaxe durability: {self.pickaxe.durability if self.pickaxe else 'N/A'}",
            f"Rocks collected: {self.rocks}",
            f"Time played: {self.time_played // 60}m {self.time_played % 60}s",
        ]
        for i, stat in enumerate(stats):
            stat_text = stats_font.render(stat, True, WHITE)
            self.screen.blit(stat_text, (SCREEN_WIDTH // 2 - stat_text.get_width() // 2, SCREEN_HEIGHT // 2 + i * 30))


        pygame.display.flip()

        # Wait for 2 seconds
        pygame.time.wait(2000)

        exit_text = stats_font.render("<press any key to exit the game>", True, GREEN)
        self.screen.blit(exit_text, (SCREEN_WIDTH // 2 - exit_text.get_width() // 2, SCREEN_HEIGHT - 40))
        pygame.display.flip()

        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    waiting = False
                elif event.type == pygame.KEYDOWN:
                    # Only exit if a key is pressed after the 3-second wait
                    waiting = False

        return False

    def run(self):
        self.start_time = pygame.time.get_ticks() - self.time_played * 1000
        running = True
        while running:
            self.time_played = (pygame.time.get_ticks() - self.start_time) // 1000
            running = self.handle_events()
            if self.player_hp <= 0:
                running = self.render_game_over()
            self.update()
            self.update_fog_of_war()
            self.check_level_cleared()
            self.render()
            self.clock.tick(FPS)
        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    game = Game()
    game.run()


