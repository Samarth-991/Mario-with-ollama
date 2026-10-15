"""
Mini Super Mario-style platformer in Python + Pygame.
Architecture follows the Medium guide: Mario / Goomba / Coin classes,
side-scrolling camera, Rect collisions, parallax background, score/lives/flag.

Run:   pip install pygame && python mario_game.py
Keys:  Left/Right or A/D = move | Space/Up/W = jump | R = restart | Esc = quit
"""
import math
import sys

import pygame

pygame.init()

# ----------------------------------------------------------------- constants
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
FPS = 60
LEVEL_WIDTH = 4000
GROUND_Y = 520
TILE = 40

GRAVITY = 0.6
MAX_FALL = 14
ACCEL = 0.6
FRICTION = 0.8
MAX_SPEED = 5.5
JUMP_POWER = 14.5

SKY_BLUE = (92, 148, 252)
MARIO_RED = (230, 30, 30)
MARIO_BLUE = (30, 60, 220)
SKIN = (255, 220, 177)
BRICK_RED = (205, 92, 92)
BRICK_DARK = (140, 60, 60)
PIPE_GREEN = (0, 150, 0)
PIPE_DARK = (0, 100, 0)
GROUND_BROWN = (150, 90, 40)
GRASS_GREEN = (80, 190, 70)
COIN_YELLOW = (255, 215, 0)
QUESTION_YELLOW = (240, 180, 40)
USED_GRAY = (130, 110, 90)
GOOMBA_BROWN = (140, 80, 30)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


# ------------------------------------------------------------------ platforms
class Platform:
    """kind: 'ground' | 'brick' | 'pipe' | 'question' | 'used'"""

    def __init__(self, x, y, w, h, kind="brick"):
        self.rect = pygame.Rect(x, y, w, h)
        self.kind = kind
        self.bump = 0  # little hop animation when hit from below

    def draw(self, screen, camera_x):
        r = self.rect.move(-camera_x, 0)
        if r.right < 0 or r.left > SCREEN_WIDTH:
            return
        if self.kind == "ground":
            pygame.draw.rect(screen, GROUND_BROWN, r)
            pygame.draw.rect(screen, GRASS_GREEN, (r.x, r.y, r.w, 10))
            for gx in range(r.x - (r.x % TILE), r.right, TILE):
                pygame.draw.line(screen, (110, 65, 25), (gx, r.y + 10), (gx, r.bottom))
        elif self.kind == "pipe":
            body = pygame.Rect(r.x + 4, r.y + 24, r.w - 8, r.h - 24)
            lip = pygame.Rect(r.x, r.y, r.w, 26)
            pygame.draw.rect(screen, PIPE_GREEN, body)
            pygame.draw.rect(screen, PIPE_DARK, body, 3)
            pygame.draw.rect(screen, PIPE_GREEN, lip)
            pygame.draw.rect(screen, PIPE_DARK, lip, 3)
            pygame.draw.rect(screen, (120, 230, 120), (lip.x + 8, lip.y + 5, 6, 16))
        else:
            r = r.move(0, -int(self.bump))
            if self.kind == "brick":
                pygame.draw.rect(screen, BRICK_RED, r)
                pygame.draw.rect(screen, BRICK_DARK, r, 2)
                pygame.draw.line(screen, BRICK_DARK, (r.x, r.centery), (r.right, r.centery), 2)
                pygame.draw.line(screen, BRICK_DARK, (r.centerx, r.y), (r.centerx, r.centery), 2)
                pygame.draw.line(screen, BRICK_DARK, (r.x + r.w // 4, r.centery), (r.x + r.w // 4, r.bottom), 2)
                pygame.draw.line(screen, BRICK_DARK, (r.x + 3 * r.w // 4, r.centery), (r.x + 3 * r.w // 4, r.bottom), 2)
            elif self.kind == "question":
                pulse = 20 * (1 + math.sin(pygame.time.get_ticks() / 150)) / 2
                pygame.draw.rect(screen, (240, 160 + pulse, 40), r)
                pygame.draw.rect(screen, (150, 90, 20), r, 3)
                font = pygame.font.SysFont(None, 34)
                q = font.render("?", True, WHITE)
                screen.blit(q, q.get_rect(center=r.center))
            else:  # used
                pygame.draw.rect(screen, USED_GRAY, r)
                pygame.draw.rect(screen, (90, 75, 60), r, 3)
        if self.bump > 0:
            self.bump = max(0, self.bump - 1.5)


# ---------------------------------------------------------------------- Mario
class Mario:
    def __init__(self, x, y):
        self.spawn = (x, y)
        self.w, self.h = 28, 36
        self.reset()

    def reset(self):
        self.x, self.y = self.spawn
        self.vel_x = self.vel_y = 0.0
        self.on_ground = False
        self.facing = 1
        self.prev_bottom = self.y + self.h
        self.invincible = 0  # frames of post-hit invincibility

    @property
    def rect(self):
        return pygame.Rect(round(self.x), round(self.y), self.w, self.h)

    def update(self, keys, platforms, game):
        left = keys[pygame.K_LEFT] or keys[pygame.K_a]
        right = keys[pygame.K_RIGHT] or keys[pygame.K_d]
        jump = keys[pygame.K_SPACE] or keys[pygame.K_UP] or keys[pygame.K_w]

        # momentum-based horizontal movement
        if left and not right:
            self.vel_x -= ACCEL
            self.facing = -1
        elif right and not left:
            self.vel_x += ACCEL
            self.facing = 1
        else:
            self.vel_x *= FRICTION
            if abs(self.vel_x) < 0.1:
                self.vel_x = 0
        self.vel_x = max(-MAX_SPEED, min(MAX_SPEED, self.vel_x))

        # jumping (variable height: release early = shorter jump)
        if jump and self.on_ground:
            self.vel_y = -JUMP_POWER
            self.on_ground = False
        if not jump and self.vel_y < -4:
            self.vel_y = -4

        self.prev_bottom = self.y + self.h

        # --- horizontal move + collide
        self.x += self.vel_x
        self.x = max(0, min(LEVEL_WIDTH - self.w, self.x))
        for p in platforms:
            if self.rect.colliderect(p.rect):
                if self.vel_x > 0:
                    self.x = p.rect.left - self.w
                elif self.vel_x < 0:
                    self.x = p.rect.right
                self.vel_x = 0

        # --- vertical move + collide
        self.vel_y = min(self.vel_y + GRAVITY, MAX_FALL)
        self.y += self.vel_y
        self.on_ground = False
        for p in platforms:
            if self.rect.colliderect(p.rect):
                if self.vel_y > 0:
                    self.y = p.rect.top - self.h
                    self.on_ground = True
                elif self.vel_y < 0:
                    self.y = p.rect.bottom
                    game.bump_block(p)
                self.vel_y = 0

        if self.invincible > 0:
            self.invincible -= 1

    def draw(self, screen, camera_x):
        if self.invincible and (self.invincible // 4) % 2:
            return  # blink while invincible
        x = round(self.x) - camera_x
        y = round(self.y)
        f = self.facing

        def box(color, ox, oy, w, h):
            ox = ox if f == 1 else self.w - ox - w
            pygame.draw.rect(screen, color, (x + ox, y + oy, w, h))

        walk = int(abs(self.vel_x) > 0.5 and self.on_ground and (pygame.time.get_ticks() // 100) % 2)
        box(MARIO_RED, 2, 0, 24, 8)        # cap
        box(MARIO_RED, 14, 4, 14, 6)       # cap brim
        box(SKIN, 4, 8, 20, 10)            # face
        box(BLACK, 16, 10, 3, 4)           # eye
        box(BLACK, 8, 15, 14, 3)           # moustache
        box(MARIO_RED, 4, 18, 20, 8)       # shirt
        box(MARIO_BLUE, 4, 24, 20, 8)      # overalls
        box(SKIN, 0, 20, 5, 6)             # hand
        box(MARIO_BLUE, 4 + walk * 4, 30, 8, 6)   # legs (animated)
        box((90, 50, 20), 12 - walk * 4, 32, 10, 4)  # boots


# --------------------------------------------------------------------- Goomba
class Goomba:
    def __init__(self, x, y):
        self.x, self.y = float(x), float(y)
        self.w, self.h = 32, 30
        self.vel_x = -1.2
        self.vel_y = 0.0
        self.alive = True
        self.squish = 0

    @property
    def rect(self):
        return pygame.Rect(round(self.x), round(self.y), self.w, self.h)

    def _ground_ahead(self, platforms):
        probe_x = self.rect.right + 2 if self.vel_x > 0 else self.rect.left - 2
        probe = pygame.Rect(probe_x, self.rect.bottom + 2, 2, 4)
        return any(probe.colliderect(p.rect) for p in platforms)

    def update(self, platforms):
        if not self.alive:
            self.squish = max(0, self.squish - 1)
            return
        # gravity
        self.vel_y = min(self.vel_y + GRAVITY, MAX_FALL)
        self.y += self.vel_y
        grounded = False
        for p in platforms:
            if self.rect.colliderect(p.rect) and self.vel_y > 0:
                self.y = p.rect.top - self.h
                self.vel_y = 0
                grounded = True
        # patrol: turn at walls and platform edges
        self.x += self.vel_x
        hit_wall = False
        for p in platforms:
            if self.rect.colliderect(p.rect):
                hit_wall = True
                self.x = p.rect.left - self.w if self.vel_x > 0 else p.rect.right
        if hit_wall or (grounded and not self._ground_ahead(platforms)):
            self.vel_x = -self.vel_x
        if self.y > SCREEN_HEIGHT + 200:
            self.alive = False

    def draw(self, screen, camera_x):
        x, y = round(self.x) - camera_x, round(self.y)
        if x < -50 or x > SCREEN_WIDTH + 50:
            return
        if not self.alive:
            if self.squish:
                pygame.draw.ellipse(screen, GOOMBA_BROWN, (x, y + 20, self.w, 10))
            return
        pygame.draw.ellipse(screen, GOOMBA_BROWN, (x, y, self.w, 24))        # head
        pygame.draw.rect(screen, SKIN, (x + 8, y + 18, 16, 8))               # body
        step = (pygame.time.get_ticks() // 180) % 2
        pygame.draw.rect(screen, (60, 30, 10), (x + (2 if step else 0), y + 24, 12, 6))  # feet
        pygame.draw.rect(screen, (60, 30, 10), (x + 18 - (2 if step else 0), y + 24, 12, 6))
        pygame.draw.rect(screen, WHITE, (x + 7, y + 8, 7, 7))                # eyes
        pygame.draw.rect(screen, WHITE, (x + 18, y + 8, 7, 7))
        pygame.draw.rect(screen, BLACK, (x + 10, y + 10, 3, 4))
        pygame.draw.rect(screen, BLACK, (x + 19, y + 10, 3, 4))
        pygame.draw.line(screen, BLACK, (x + 6, y + 6), (x + 14, y + 9), 2)  # angry brows
        pygame.draw.line(screen, BLACK, (x + 26, y + 6), (x + 18, y + 9), 2)


# ----------------------------------------------------------------------- Coin
class Coin:
    def __init__(self, x, y, popup=False):
        self.x, self.y = x, y
        self.w, self.h = 20, 28
        self.rotation = 0.0
        self.collected = False
        self.popup = popup          # coin from a ? block: rises then vanishes
        self.timer = 40

    @property
    def rect(self):
        return pygame.Rect(self.x, self.y, self.w, self.h)

    def update(self):
        self.rotation += 0.2
        if self.popup:
            self.y -= 3
            self.timer -= 1
            if self.timer <= 0:
                self.collected = True

    def draw(self, screen, camera_x):
        if self.collected:
            return
        x = self.x - camera_x
        scale = abs(math.sin(self.rotation))
        width = max(2, int(self.w * scale))
        r = (x + (self.w - width) // 2, self.y, width, self.h)
        pygame.draw.ellipse(screen, COIN_YELLOW, r)
        pygame.draw.ellipse(screen, (200, 150, 0), r, 2)


# ----------------------------------------------------------------------- Game
class Game:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Mini Mario - Python")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 34)
        self.big_font = pygame.font.SysFont(None, 80)
        self.reset()

    # ---- level --------------------------------------------------------
    def build_level(self):
        P = []
        # ground with gaps
        for x0, x1 in [(0, 1200), (1360, 2280), (2440, LEVEL_WIDTH)]:
            P.append(Platform(x0, GROUND_Y, x1 - x0, SCREEN_HEIGHT - GROUND_Y, "ground"))
        # pipes
        for px, ph in [(600, 80), (1650, 120), (2700, 80), (3300, 120)]:
            P.append(Platform(px, GROUND_Y - ph, 70, ph, "pipe"))
        # floating bricks / question blocks
        def row(x, y, kinds):
            for i, k in enumerate(kinds):
                P.append(Platform(x + i * TILE, y, TILE, TILE, k))
        row(300, 390, ["brick", "question", "brick", "question", "brick"])
        row(900, 330, ["brick", "brick", "brick"])
        row(1180, 390, ["question"])
        row(1900, 390, ["brick", "question", "brick"])
        row(1960, 270, ["question"])
        row(2250, 360, ["brick", "brick"])
        row(2980, 390, ["brick", "question", "brick", "question"])
        row(3500, 400, ["brick", "brick", "brick", "brick"])
        self.platforms = P

        self.goombas = [Goomba(x, GROUND_Y - 30) for x in
                        (500, 800, 1000, 1500, 1800, 2100, 2600, 3000, 3450, 3650)]

        coins = []
        for x in range(380, 520, 40):
            coins.append(Coin(x + 10, 340))
        for x in range(900, 1020, 40):
            coins.append(Coin(x + 10, 280))
        for i in range(5):  # coin arc over first gap
            coins.append(Coin(1130 + i * 45, 400 - int(70 * math.sin(i / 4 * math.pi))))
        for x in range(1750, 1900, 45):
            coins.append(Coin(x, 440))
        for i in range(5):  # arc over second gap
            coins.append(Coin(2230 + i * 45, 400 - int(70 * math.sin(i / 4 * math.pi))))
        for x in range(3050, 3250, 45):
            coins.append(Coin(x, 440))
        self.coins = coins

        self.flag_x = 3850
        self.flag_pole = pygame.Rect(self.flag_x, GROUND_Y - 300, 8, 300)

    def reset(self):
        self.build_level()
        self.mario = Mario(80, GROUND_Y - 36)
        self.score = 0
        self.lives = 3
        self.coin_count = 0
        self.camera_x = 0
        self.state = "playing"  # playing | won | gameover

    # ---- interactions -------------------------------------------------
    def bump_block(self, p):
        """Called when Mario's head hits a platform from below."""
        if p.kind in ("brick", "question", "used"):
            p.bump = 8
        if p.kind == "question":
            p.kind = "used"
            self.coins.append(Coin(p.rect.centerx - 10, p.rect.top - 34, popup=True))
            self.score += 100
            self.coin_count += 1

    def hurt_mario(self):
        if self.mario.invincible:
            return
        self.lives -= 1
        if self.lives <= 0:
            self.state = "gameover"
            return
        self.respawn()

    def respawn(self):
        # respawn at last safe checkpoint (start of the current section)
        cx = self.mario.x
        checkpoint = max([x for x in (80, 1300, 2400) if x <= cx] or [80])
        self.mario.reset()
        self.mario.x, self.mario.y = checkpoint, GROUND_Y - 36
        self.mario.invincible = 120

    def handle_collisions(self):
        m = self.mario
        mr = m.rect

        for coin in self.coins:
            if not coin.collected and not coin.popup and mr.colliderect(coin.rect):
                coin.collected = True
                self.score += 100
                self.coin_count += 1

        for g in self.goombas:
            if g.alive and mr.colliderect(g.rect):
                stomp = m.vel_y > 0 and m.prev_bottom <= g.rect.top + 10
                if stomp:
                    g.alive = False
                    g.squish = 30
                    m.vel_y = -8
                    self.score += 200
                else:
                    self.hurt_mario()
                    return

        if mr.colliderect(self.flag_pole):
            self.state = "won"
            self.score += max(0, self.lives) * 500

        if m.y > SCREEN_HEIGHT + 50:  # fell into a gap
            self.hurt_mario()

    def update_camera(self):
        self.camera_x = int(self.mario.x - SCREEN_WIDTH // 2)
        self.camera_x = max(0, min(LEVEL_WIDTH - SCREEN_WIDTH, self.camera_x))

    def update(self, keys):
        if self.state != "playing":
            return
        self.mario.update(keys, self.platforms, self)
        for g in self.goombas:
            g.update(self.platforms)
        for c in self.coins:
            c.update()
        self.handle_collisions()
        self.update_camera()

    # ---- drawing ------------------------------------------------------
    def draw_cloud(self, x, y):
        for dx, dy, r in [(0, 0, 22), (24, -10, 28), (52, 0, 22), (26, 8, 22)]:
            pygame.draw.circle(self.screen, WHITE, (int(x + dx), int(y + dy)), r)

    def draw_hill(self, base_x, w, h):
        x = base_x - self.camera_x * 0.3
        pts = [(x, GROUND_Y), (x + w / 2, GROUND_Y - h), (x + w, GROUND_Y)]
        pygame.draw.polygon(self.screen, (60, 170, 70), pts)

    def draw_background(self):
        self.screen.fill(SKY_BLUE)
        for i in range(8):
            self.draw_hill(i * 520 - 100, 360, 120 + (i % 3) * 30)
        for i in range(10):
            x = (i * 260 - self.camera_x * 0.5) % (SCREEN_WIDTH + 200) - 100
            self.draw_cloud(x, 70 + (i % 3) * 35)

    def draw_flag(self):
        x = self.flag_x - self.camera_x
        top = GROUND_Y - 300
        pygame.draw.rect(self.screen, (230, 230, 230), (x, top, 8, 300))
        pygame.draw.circle(self.screen, COIN_YELLOW, (x + 4, top - 6), 9)
        wave = int(3 * math.sin(pygame.time.get_ticks() / 200))
        pygame.draw.polygon(self.screen, (40, 200, 60),
                            [(x + 8, top + 8), (x + 62, top + 28 + wave), (x + 8, top + 48)])
        pygame.draw.rect(self.screen, (90, 90, 90), (x - 14, GROUND_Y - 20, 36, 20))

    def draw_ui(self):
        txt = f"SCORE {self.score:06d}    COINS x{self.coin_count:02d}    LIVES x{self.lives}"
        shadow = self.font.render(txt, True, BLACK)
        label = self.font.render(txt, True, WHITE)
        self.screen.blit(shadow, (22, 22))
        self.screen.blit(label, (20, 20))

    def draw_center(self, title, sub, color):
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 140))
        self.screen.blit(overlay, (0, 0))
        t = self.big_font.render(title, True, color)
        s = self.font.render(sub, True, WHITE)
        self.screen.blit(t, t.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 30)))
        self.screen.blit(s, s.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 30)))

    def draw(self):
        self.draw_background()
        cx = self.camera_x
        for p in self.platforms:
            p.draw(self.screen, cx)
        self.draw_flag()
        for c in self.coins:
            c.draw(self.screen, cx)
        for g in self.goombas:
            g.draw(self.screen, cx)
        self.mario.draw(self.screen, cx)
        self.draw_ui()
        if self.state == "won":
            self.draw_center("LEVEL CLEAR!", f"Final score {self.score}  -  press R to play again", COIN_YELLOW)
        elif self.state == "gameover":
            self.draw_center("GAME OVER", "Press R to try again", (255, 80, 80))

    # ---- main loop ----------------------------------------------------
    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        self.reset()
            self.update(pygame.key.get_pressed())
            self.draw()
            pygame.display.flip()
            self.clock.tick(FPS)
        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    Game().run()
