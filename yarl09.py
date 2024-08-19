#################################################################################
#                                                                               #
# YARL is an endless roguelike game, written in Python3 plus Pygame library     #
# Copyright (C) 2024 by paws9678 @ Discord                                      #
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
import json
import os
from dataclasses import dataclass
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
DARK_GRAY = (50, 50, 50)
WHITE = (255, 255, 255)
DARK_BLUE = (0, 0, 128)
GRAY = (128, 128, 128)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# Tile types
FLOOR = 0
WALL = 1
STAIRS_DOWN = 2
STAIRS_UP = 3
TRAP = 4
REVEALED_TRAP = 5


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

    @property
    def size_factor(self):
        return max(0.05, min(1.0, self.current_hp / self.max_hp))

class Level:
    def __init__(self, level_number: int):
        self.level_number = level_number
        self.grid = [[WALL for _ in range(LEVEL_WIDTH)] for _ in range(LEVEL_HEIGHT)]
        self.random = random.Random(level_number)
        self.stairs_up_position = None
        self.stairs_down_position = None
        self.revealed_traps: Dict[Tuple[int, int], bool] = {}
        self.monsters: List[Monster] = []
        self.monster_damage = 10 + (level_number - 1)  # Damage scales with level
        self.generate()


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



        # Place more monsters
        num_monsters = self.random.randint(18, 48)  # Doubled from (9, 24)
        base_hp = 50
        level_hp_increase = max(0, self.level_number - 1) * 5 // 100 * base_hp
        monster_hp = base_hp + level_hp_increase

        for _ in range(num_monsters):
            if empty_tiles:
                monster_pos = self.random.choice(empty_tiles)
                self.monsters.append(Monster(monster_pos[0], monster_pos[1], monster_hp, monster_hp))
                empty_tiles.remove(monster_pos)

        # Place more traps
        num_traps = self.random.randint(7, 15)  # Increased number of traps, from 3-15 to 7-15
        for _ in range(num_traps):
            if empty_tiles:
                trap = self.random.choice(empty_tiles)
                self.grid[trap[1]][trap[0]] = TRAP
                empty_tiles.remove(trap)


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

    def reveal_trap(self, x: int, y: int):
        self.revealed_traps[(x, y)] = True

    def move_monsters(self, player_x, player_y):
        attacking_monster = None
        for monster in self.monsters:
            dx = player_x - monster.x
            dy = player_y - monster.y
            distance = max(abs(dx), abs(dy))  # Chebyshev distance

            if distance <= 1:  # Monster is adjacent (including diagonally) to player
                attacking_monster = monster
                break
            elif distance <= 8:  # Monster is within 8 steps of the player
                # Move towards the player, preferring diagonal movement
                move_x = 1 if dx > 0 else -1 if dx < 0 else 0
                move_y = 1 if dy > 0 else -1 if dy < 0 else 0
                
                new_x, new_y = monster.x + move_x, monster.y + move_y

                if not self.is_blocked(new_x, new_y):
                    monster.x, monster.y = new_x, new_y
                elif not self.is_blocked(monster.x + move_x, monster.y):
                    monster.x += move_x
                elif not self.is_blocked(monster.x, monster.y + move_y):
                    monster.y += move_y
            else:  # Monster is far from the player, move with increasing likelihood towards the player
                move_towards_player = self.random.random() < (1 / (distance + 1))  # Likelihood increases as distance decreases
                if move_towards_player:
                    move_x = 1 if dx > 0 else -1 if dx < 0 else 0
                    move_y = 1 if dy > 0 else -1 if dy < 0 else 0
                    # Prioritize the direction with the larger difference
                    if abs(dx) > abs(dy):
                        new_x, new_y = monster.x + move_x, monster.y
                        if not self.is_blocked(new_x, new_y):
                            monster.x = new_x
                        elif not self.is_blocked(monster.x, monster.y + move_y):
                            monster.y += move_y
                    else:
                        new_x, new_y = monster.x, monster.y + move_y
                        if not self.is_blocked(new_x, new_y):
                            monster.y = new_y
                        elif not self.is_blocked(monster.x + move_x, monster.y):
                            monster.x += move_x
                else:
                    # Random movement
                    dx, dy = self.random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
                    new_x, new_y = monster.x + dx, monster.y + dy
                    if not self.is_blocked(new_x, new_y):
                        monster.x, monster.y = new_x, new_y

        return attacking_monster, self.monster_damage

    def is_blocked(self, x, y):
        if not (0 <= x < LEVEL_WIDTH and 0 <= y < LEVEL_HEIGHT):
            return True
        if self.grid[y][x] == WALL:
            return True
        if any(m for m in self.monsters if m.x == x and m.y == y):
            return True
        return False

    def get_quadrant(self, x, y):
        if x < LEVEL_WIDTH // 2:
            return "West" if y < LEVEL_HEIGHT // 2 else "Southwest"
        else:
            return "Northeast" if y < LEVEL_HEIGHT // 2 else "Southeast"

    def dig_tunnels(self, x, y):
        quadrant = self.get_quadrant(x, y)
        tunnel_length = 20

        if quadrant in ["West", "Southwest"]:
            # Dig east
            for i in range(1, tunnel_length + 1):
                if x + i < LEVEL_WIDTH:
                    self.grid[y][x + i] = FLOOR

        if quadrant in ["West", "Northeast"]:
            # Dig south
            for i in range(1, tunnel_length + 1):
                if y + i < LEVEL_HEIGHT:
                    self.grid[y + i][x] = FLOOR

        if quadrant in ["Northeast", "Southeast"]:
            # Dig west
            for i in range(1, tunnel_length + 1):
                if x - i >= 0:
                    self.grid[y][x - i] = FLOOR

        if quadrant in ["Southwest", "Southeast"]:
            # Dig north
            for i in range(1, tunnel_length + 1):
                if y - i >= 0:
                    self.grid[y - i][x] = FLOOR

    def handle_trap_fall(self, x, y):
        self.grid[y][x] = FLOOR  # Ensure the landing spot is a floor
        self.dig_tunnels(x, y)   # Dig tunnels from the landing spot
        return x, y  # Return the landing coordinates

class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.FULLSCREEN)
        pygame.display.set_caption("YARL - Yet Another RogueLike v0.9")
        self.clock = pygame.time.Clock()
        self.player = GameObject(LEVEL_WIDTH // 2, LEVEL_HEIGHT // 2, DARK_BLUE, 'O')
        self.current_level = 0
        self.levels = {}
        self.get_or_create_level(self.current_level)
        self.direction = "North"
        self.player_hp = 5000
        self.font = pygame.font.Font(None, 36)
        self.last_move_time = 0
        self.move_direction = None
        self.weapon = Weapon("Level 1 Sword", 10)
        self.fog_of_war = [[False for _ in range(LEVEL_WIDTH)] for _ in range(LEVEL_HEIGHT)]
        self.visibility_radius = 20  # Doubled from 10
        self.monsters_slain = 0
        self.hits_taken = 0
        self.steps_walked = 0
        self.traps_fallen = 0
        self.levels_cleared = 0
        self.start_time = pygame.time.get_ticks()
        self.time_played = 0

        self.save_file_path = os.path.join(os.path.dirname(__file__), 'savegame.json')
        if os.path.exists(self.save_file_path):
            self.show_load_option()
        else:
            self.new_game()



    def check_level_cleared(self):
        current_level = self.get_or_create_level(self.current_level)
        if not current_level.monsters:
            self.levels_cleared += 1


    def new_game(self):
        self.player = GameObject(LEVEL_WIDTH // 2, LEVEL_HEIGHT // 2, DARK_BLUE, 'O')
        self.current_level = 0
        self.levels = {}
        self.get_or_create_level(self.current_level)
        self.direction = "North"
        self.player_hp = 5000
        self.last_move_time = 0
        self.move_direction = None
        self.weapon = Weapon("Level 1 Sword", 10)
        self.fog_of_war = [[False for _ in range(LEVEL_WIDTH)] for _ in range(LEVEL_HEIGHT)]
        self.visibility_radius = 20
        self.monsters_slain = 0
        self.hits_taken = 0
        self.steps_walked = 0
        self.traps_fallen = 0
        self.levels_cleared = 0
        self.start_time = pygame.time.get_ticks()
        self.time_played = 0


    def save_game(self):
        save_data = {
            'player_x': self.player.x,
            'player_y': self.player.y,
            'player_hp': self.player_hp,
            'current_level': self.current_level,
            'monsters_slain': self.monsters_slain,
            'hits_taken': self.hits_taken,
            'steps_walked': self.steps_walked,
            'traps_fallen': self.traps_fallen,
            'levels_cleared': self.levels_cleared,
            'time_played': self.time_played,
            'weapon_name': self.weapon.name,
            'weapon_damage': self.weapon.damage,
        }
        with open(self.save_file_path, 'w') as f:
            json.dump(save_data, f)



    def load_game(self):
        try:
            with open(self.save_file_path, 'r') as f:
                save_data = json.load(f)
            self.player.x = save_data['player_x']
            self.player.y = save_data['player_y']
            self.player_hp = save_data['player_hp']
            self.current_level = save_data['current_level']
            self.monsters_slain = save_data['monsters_slain']
            self.hits_taken = save_data['hits_taken']
            self.steps_walked = save_data['steps_walked']
            self.traps_fallen = save_data['traps_fallen']
            self.levels_cleared = save_data['levels_cleared']
            self.time_played = save_data['time_played']
            self.weapon = Weapon(save_data['weapon_name'], save_data['weapon_damage'])
            self.get_or_create_level(self.current_level)
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


    def get_or_create_level(self, level_number):
        if level_number not in self.levels:
            self.levels[level_number] = Level(level_number)
        return self.levels[level_number]

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                self.save_game()
                return False
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT):
                    self.move_direction = event.key
                    self.last_move_time = pygame.time.get_ticks()
                elif event.key == pygame.K_u:
                    self.use_stairs('up')
                elif event.key == pygame.K_d:
                    self.use_stairs('down')
                elif event.key == pygame.K_a:
                    self.attack()
            if event.type == pygame.KEYUP:
                if event.key == self.move_direction:
                    self.move_direction = None
        return True


    def use_stairs(self, direction):
        current_level = self.get_or_create_level(self.current_level)
        player_tile = current_level.grid[self.player.y][self.player.x]
        
        if direction == 'up' and player_tile == STAIRS_UP:
            self.current_level -= 1
            next_level = self.get_or_create_level(self.current_level)
            if next_level.stairs_down_position:
                self.player.x, self.player.y = next_level.stairs_down_position
            else:
                self.player.x, self.player.y = LEVEL_WIDTH // 2, LEVEL_HEIGHT // 2
        elif direction == 'down' and player_tile == STAIRS_DOWN:
            self.current_level += 1
            next_level = self.get_or_create_level(self.current_level)
            if next_level.stairs_up_position:
                self.player.x, self.player.y = next_level.stairs_up_position
            else:
                # If stairs_up_position is None, place the player in the center and create stairs
                self.player.x, self.player.y = LEVEL_WIDTH // 2, LEVEL_HEIGHT // 2
                next_level.stairs_up_position = (self.player.x, self.player.y)
                next_level.grid[self.player.y][self.player.x] = STAIRS_UP
        
        self.update_fog_of_war()


    def move_player(self, dx, dy):
        new_x, new_y = self.player.x + dx, self.player.y + dy
        current_level = self.get_or_create_level(self.current_level)
        if 0 <= new_x < LEVEL_WIDTH and 0 <= new_y < LEVEL_HEIGHT:
            if current_level.grid[new_y][new_x] != WALL:
                monster = next((m for m in current_level.monsters if m.x == new_x and m.y == new_y), None)
                if monster:
                    self.attack_monster(monster)
                else:
                    self.player.x, self.player.y = new_x, new_y
                    self.steps_walked += 1  # Increment steps walked
                    self.player_hp = min(5000, self.player_hp + 1)  # Heal 1 HP per step, max 5000

                    if current_level.grid[new_y][new_x] == TRAP:
                        current_level.reveal_trap(new_x, new_y)
                        self.player_hp -= 100  # 2% of 5000 HP
                        self.player_hp = max(0, self.player_hp)  # Ensure HP doesn't go below 0
                        self.hits_taken += 1  # Increment hits taken
                        self.traps_fallen += 1  # Increment traps fallen
                        
                        # Fall to the next level at the same x, y coordinates
                        self.current_level += 1
                        next_level = self.get_or_create_level(self.current_level)
                        
                        # Handle trap fall in the next level
                        self.player.x, self.player.y = next_level.handle_trap_fall(new_x, new_y)
                
                self.handle_monster_turn()
        self.update_fog_of_war()


    def attack_monster(self, monster):
        monster.current_hp -= self.weapon.damage
        if monster.current_hp <= 0:
            current_level = self.get_or_create_level(self.current_level)
            current_level.monsters.remove(monster)
            self.monsters_slain += 1  # Increment monsters slain
        self.handle_monster_turn()


    def handle_monster_turn(self):
        current_level = self.get_or_create_level(self.current_level)
        attacking_monster, monster_damage = current_level.move_monsters(self.player.x, self.player.y)
        if attacking_monster:
            dx = abs(self.player.x - attacking_monster.x)
            dy = abs(self.player.y - attacking_monster.y)
            if dx <= 1 and dy <= 1:  # Allow diagonal attacks
                self.player_hp -= monster_damage
                self.player_hp = max(0, self.player_hp)
                self.hits_taken += 1  # Increment hits taken


    def attack(self):
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

    def update(self):
        current_time = pygame.time.get_ticks()
        if self.move_direction:
            if current_time - self.last_move_time >= MOVE_REPEAT_INTERVAL:
                self.last_move_time = current_time
                if self.move_direction == pygame.K_UP:
                    self.move_player(0, -1)
                    self.direction = "North"
                elif self.move_direction == pygame.K_DOWN:
                    self.move_player(0, 1)
                    self.direction = "South"
                elif self.move_direction == pygame.K_LEFT:
                    self.move_player(-1, 0)
                    self.direction = "West"
                elif self.move_direction == pygame.K_RIGHT:
                    self.move_player(1, 0)
                    self.direction = "East"

    def update_fog_of_war(self):
        current_level = self.get_or_create_level(self.current_level)
        for y in range(LEVEL_HEIGHT):
            for x in range(LEVEL_WIDTH):
                distance = max(abs(self.player.x - x), abs(self.player.y - y))
                if distance <= self.visibility_radius:
                    self.fog_of_war[y][x] = True

    def render(self):
        self.screen.fill(BLACK)
        current_level = self.get_or_create_level(self.current_level)
        
        # Calculate the visible area
        visible_width = SCREEN_WIDTH // TILE_SIZE
        visible_height = (SCREEN_HEIGHT - 40) // TILE_SIZE
        start_x = max(0, min(self.player.x - visible_width // 2, LEVEL_WIDTH - visible_width))
        start_y = max(0, min(self.player.y - visible_height // 2, LEVEL_HEIGHT - visible_height))
        
        for y in range(visible_height):
            for x in range(visible_width):
                world_x = start_x + x
                world_y = start_y + y
                if 0 <= world_x < LEVEL_WIDTH and 0 <= world_y < LEVEL_HEIGHT:
                    if self.fog_of_war[world_y][world_x]:
                        tile = current_level.grid[world_y][world_x]
                        color = BLACK if tile == WALL else DARK_GRAY
                        pygame.draw.rect(self.screen, color, (x * TILE_SIZE, y * TILE_SIZE + 40, TILE_SIZE, TILE_SIZE))
                        
                        if tile == STAIRS_DOWN:
                            pygame.draw.line(self.screen, GRAY, (x * TILE_SIZE, y * TILE_SIZE + 40), 
                                             ((x + 1) * TILE_SIZE, (y + 1) * TILE_SIZE + 40), 2)
                        elif tile == STAIRS_UP:
                            pygame.draw.line(self.screen, GRAY, (x * TILE_SIZE, (y + 1) * TILE_SIZE + 40), 
                                             ((x + 1) * TILE_SIZE, y * TILE_SIZE + 40), 2)
                        elif (world_x, world_y) in current_level.revealed_traps:
                            pygame.draw.line(self.screen, RED, (x * TILE_SIZE, y * TILE_SIZE + 40), 
                                             ((x + 1) * TILE_SIZE, (y + 1) * TILE_SIZE + 40), 2)
                            pygame.draw.line(self.screen, RED, (x * TILE_SIZE, (y + 1) * TILE_SIZE + 40), 
                                             ((x + 1) * TILE_SIZE, y * TILE_SIZE + 40), 2)
                    else:
                        # Draw unexplored areas
                        pygame.draw.rect(self.screen, (20, 20, 20), (x * TILE_SIZE, y * TILE_SIZE + 40, TILE_SIZE, TILE_SIZE))

        for monster in current_level.monsters:
            if (start_x <= monster.x < start_x + visible_width and 
                start_y <= monster.y < start_y + visible_height and
                self.fog_of_war[monster.y][monster.x]):
                screen_x = (monster.x - start_x) * TILE_SIZE
                screen_y = (monster.y - start_y) * TILE_SIZE + 40
                size = int(TILE_SIZE * max(0.05, monster.current_hp / monster.max_hp))
                center_x = screen_x + TILE_SIZE // 2
                center_y = screen_y + TILE_SIZE // 2
                pygame.draw.polygon(self.screen, GREEN, 
                                    [(center_x, center_y - size // 2),
                                     (center_x - size // 2, center_y + size // 2),
                                     (center_x + size // 2, center_y + size // 2)])

        # Render player
        player_screen_x = (self.player.x - start_x) * TILE_SIZE
        player_screen_y = (self.player.y - start_y) * TILE_SIZE + 40
        pygame.draw.circle(self.screen, self.player.color, 
                           (player_screen_x + TILE_SIZE // 2, 
                            player_screen_y + TILE_SIZE // 2), TILE_SIZE // 2)

        # Add a mini-map
        mini_map_size = 100
        mini_map_surface = pygame.Surface((mini_map_size, mini_map_size))
        mini_map_surface.fill((50, 50, 50))
        for y in range(LEVEL_HEIGHT):
            for x in range(LEVEL_WIDTH):
                if self.fog_of_war[y][x]:
                    color = (0, 0, 0) if current_level.grid[y][x] == WALL else (100, 100, 100)
                    pygame.draw.rect(mini_map_surface, color, (x * mini_map_size // LEVEL_WIDTH, y * mini_map_size // LEVEL_HEIGHT, 
                                     max(1, mini_map_size // LEVEL_WIDTH), max(1, mini_map_size // LEVEL_HEIGHT)))
        
        # Draw player position on mini-map
        pygame.draw.rect(mini_map_surface, (0, 0, 255), (self.player.x * mini_map_size // LEVEL_WIDTH, 
                         self.player.y * mini_map_size // LEVEL_HEIGHT, 2, 2))
        
        self.screen.blit(mini_map_surface, (SCREEN_WIDTH - mini_map_size - 10, SCREEN_HEIGHT - mini_map_size - 10))

        # Render HUD
        hud_text = f"Level: {self.current_level + 1} | X: {self.player.x} Y: {self.player.y} | Direction: {self.direction} | HP: {self.player_hp} | Weapon: {self.weapon.name} (DMG: {self.weapon.damage})"
        hud_surface = self.font.render(hud_text, True, WHITE)
        self.screen.blit(hud_surface, (10, 10))

        # Render HP bar
        pygame.draw.rect(self.screen, RED, (SCREEN_WIDTH - 210, 10, 200, 20))
        pygame.draw.rect(self.screen, (0, 255, 0), (SCREEN_WIDTH - 210, 10, min(200, self.player_hp // 25), 20))
        
        pygame.display.flip()


    def render_game_over(self):
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        overlay.set_alpha(180)
        overlay.fill((0, 0, 0))
        self.screen.blit(overlay, (0, 0))

        font = pygame.font.Font(None, 48)
        game_over_text = font.render("G A M E   O V E R  --  You have died, better luck next time.", True, WHITE)
        self.screen.blit(game_over_text, (SCREEN_WIDTH // 2 - game_over_text.get_width() // 2, SCREEN_HEIGHT // 2 - 100))

        stats_font = pygame.font.Font(None, 36)
        stats = [
            f"Levels entered: {self.current_level + 1}",
            f"Monsters slain: {self.monsters_slain}",
            f"Hits taken: {self.hits_taken}",
            f"Steps walked: {self.steps_walked}",
            f"Traps fallen: {self.traps_fallen}",
            f"Levels cleared: {self.levels_cleared}",
            f"Time played: {self.time_played // 60}m {self.time_played % 60}s"
        ]
        for i, stat in enumerate(stats):
            stat_text = stats_font.render(stat, True, WHITE)
            self.screen.blit(stat_text, (SCREEN_WIDTH // 2 - stat_text.get_width() // 2, SCREEN_HEIGHT // 2 + i * 40))

        exit_text = stats_font.render("<press any key to exit the game>", True, WHITE)
        self.screen.blit(exit_text, (SCREEN_WIDTH // 2 - exit_text.get_width() // 2, SCREEN_HEIGHT - 100))

        pygame.display.flip()

        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or event.type == pygame.KEYDOWN:
                    waiting = False
                    return False
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



