import os
import time
import copy
from typing import List, Tuple, Optional

class Connect4:
    """
    Standard Connect 4 Game Engine.
    Board: 6 rows x 7 cols.
    0 = Empty, 1 = Red, 2 = Yellow.
    """
    
    ROWS: int = 6
    COLS: int = 7
    EMPTY: int = 0
    RED: int = 1
    YELLOW: int = 2

    TOKEN_RED: str = "🔴"
    TOKEN_YELLOW: str = "🟡"
    TOKEN_EMPTY: str = "⚪"
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        """Resets the board to empty state."""
        # 6 rows, 7 cols
        self.board = [[self.EMPTY for _ in range(self.COLS)] for _ in range(self.ROWS)]
        self.turn = self.RED
        self.moves_made = 0
        self.last_move: Optional[Tuple[int, int]] = None
        self.winner: Optional[int] = None
        
    def get_state_hash(self) -> str:
        """Unique string hash of the board state for MCTS."""
        # Simple flattening
        flat = "".join(str(cell) for row in self.board for cell in row)
        return f"{flat}:{self.turn}"
        
    def is_terminal(self) -> bool:
        """True if someone won or draw."""
        return self.winner is not None or self.check_draw()
        
    def get_result(self, player_id: int) -> float:
        """
        Returns 1.0 if player_id won, 0.0 if lost, 0.5 if draw.
        (MCTS value range usually 0-1)
        """
        if self.winner == player_id:
            return 1.0
        elif self.winner is not None:
             # Opponent won
            return 0.0
        elif self.check_draw():
            return 0.5
        return 0.0 # Game not over

        
    def copy(self) -> 'Connect4':
        """Deep copy for lookahead."""
        new_game = Connect4()
        new_game.board = copy.deepcopy(self.board)
        new_game.turn = self.turn
        new_game.moves_made = self.moves_made
        new_game.last_move = self.last_move
        new_game.winner = self.winner
        return new_game
        
    def get_valid_moves(self) -> List[int]:
        """Returns list of column indices (0-6) that are not full."""
        moves = []
        if self.winner is not None:
            return moves
            
        for c in range(self.COLS):
            if self.board[0][c] == self.EMPTY:
                moves.append(c)
        return moves
        
    def drop_token(self, col: int, player_id: int) -> Tuple[int, int]:
        """
        Drops a token into the specified column.
        Returns (row, col) where it landed.
        Raises ValueError if column full or invalid.
        """
        if col < 0 or col >= self.COLS:
            raise ValueError(f"Invalid column: {col}")
            
        # Find the lowest empty row
        landing_row = -1
        for r in range(self.ROWS - 1, -1, -1):
            if self.board[r][col] == self.EMPTY:
                self.board[r][col] = player_id
                landing_row = r
                break
                
        if landing_row == -1:
            raise ValueError(f"Column {col} is full!")
            
        self.last_move = (landing_row, col)
        self.moves_made += 1
        
        # Check for win immediately
        if self.check_win(player_id):
            self.winner = player_id
            
        return (landing_row, col)

    def check_win(self, player_id: int) -> bool:
        """Checks if player_id has won (4 in a row)."""
        # Horizontal
        for r in range(self.ROWS):
            for c in range(self.COLS - 3):
                if all(self.board[r][c+i] == player_id for i in range(4)):
                    return True
                    
        # Vertical
        for r in range(self.ROWS - 3):
            for c in range(self.COLS):
                if all(self.board[r+i][c] == player_id for i in range(4)):
                    return True
                    
        # Diagonal /
        for r in range(3, self.ROWS):
            for c in range(self.COLS - 3):
                if all(self.board[r-i][c+i] == player_id for i in range(4)):
                    return True
                    
        # Diagonal \
        for r in range(self.ROWS - 3):
            for c in range(self.COLS - 3):
                if all(self.board[r+i][c+i] == player_id for i in range(4)):
                    return True
                    
        return False
        
    def check_draw(self) -> bool:
        """Checks if board is full and no winner."""
        return self.moves_made >= (self.ROWS * self.COLS) and self.winner is None
        
    def to_string(self, labeled: bool = False) -> str:
        """Returns string representation of board."""
        rows = []
        if labeled:
            rows.append("  1  2  3  4  5  6  7")
        rows.append(" ┌──┬──┬──┬──┬──┬──┬──┐")
        for row in self.board:
            row_str = " │"
            for cell in row:
                if cell == self.RED:
                    row_str += self.TOKEN_RED
                elif cell == self.YELLOW:
                    row_str += self.TOKEN_YELLOW
                else:
                    row_str += self.TOKEN_EMPTY
                row_str += "│"
            rows.append(row_str)
            if row != self.board[-1]: # Don't add separator after last
                rows.append(" ├──┼──┼──┼──┼──┼──┼──┤")
        rows.append(" └──┴──┴──┴──┴──┴──┴──┘")
        
        return "\n".join(rows)

    def to_list_repr(self) -> str:
        """Returns list of moves for LLM context."""
        moves = []
        for r in range(self.ROWS):
            for c in range(self.COLS):
                 tok = self.board[r][c]
                 if tok != self.EMPTY:
                     color = "RED" if tok == self.RED else "YELLOW"
                     moves.append(f"{color} at Col {c+1} Row {r+1}")
        if not moves: return "Empty Board"
        return "; ".join(moves)

    def render_ascii(self, animate_drop: Optional[Tuple[int, int]] = None, speed: str = "normal", hud_data: Optional[str] = None):
        """
        Renders the board to the terminal using ANSI codes.
        animate_drop: (target_row, col) - if provided, animates the drop.
        """
        # Clear screen
        os.system('cls' if os.name == 'nt' else 'clear')
        
        # Determine delay
        delay = 0.05
        if speed == "fast": delay = 0.01
        elif speed == "slow": delay = 0.2
        
        print("\n\033[1;36m=== CONNECT 4 DUEL ===\033[0m")
        if hud_data:
            print(hud_data)
        print()
        
        # Helper to print board state
        def print_grid(floating_token_pos: Optional[Tuple[int, int]] = None):
            # Header
            print("  1  2  3  4  5  6  7")
            print(" ┌" + "──┬" * 6 + "──┐")
            
            for r in range(self.ROWS):
                row_str = " │"
                for c in range(self.COLS):
                    val = self.board[r][c]
                    
                    # Override if we are animating a drop at this position
                    if floating_token_pos and floating_token_pos == (r, c):
                         # Look at whose turn it WAS (since we already updated state in drop_token)
                         # Actually checking board value is safer as it is the committed state
                         val = self.board[r][c] 

                    sym = self.TOKEN_EMPTY
                    if val == self.RED: sym = self.TOKEN_RED
                    elif val == self.YELLOW: sym = self.TOKEN_YELLOW
                    
                    # If this is the *final* resting place of the last move, maybe highlight?
                    # For now keep simple
                    
                    row_str += sym + "│"
                print(row_str)
                if r < self.ROWS - 1:
                    print(" ├" + "──┼" * 6 + "──┤")
            print(" └" + "──┴" * 6 + "──┘")
            
        # Animation
        if animate_drop:
            target_r, target_c = animate_drop
            # Temporarily clear the target spot for animation logic if we want to show it falling
            # But the logic is easier if we just print the 'falling' token OVER the empty spots
            # Since drop_token already refilled the board, we are animating "after the fact" in a way
            # Ideally we'd animate BEFORE committing to board, but for simplicity:
            pass 
            # DDA-X Requirement: "show token drops with simple animation"
            # To do this right, we need to draw the board with the target slot EMPTY, then draw the token falling
            
            player_val = self.board[target_r][target_c]
            # Temporarily empty it
            self.board[target_r][target_c] = self.EMPTY
            
            # Drop from top to target
            for r in range(target_r + 1):
                # Clear screen reset
                os.system('cls' if os.name == 'nt' else 'clear')
                print("\n\033[1;36m=== CONNECT 4 DUEL ===\033[0m")
                if hud_data: print(hud_data)
                print()
                
                # We need to render the board, but with the falling token at (r, target_c)
                # But we can't modify self.board... so we handle it purely in print logic?
                # A simpler hack: put it in board, print, then remove
                
                if r == target_r:
                    # Final spot
                    self.board[r][target_c] = player_val
                    print_grid()
                else:
                    # Falling
                    self.board[r][target_c] = player_val # Put it here
                    print_grid()
                    self.board[r][target_c] = self.EMPTY # Remove
                    time.sleep(delay)
                    
            # Ensure it is set back
            self.board[target_r][target_c] = player_val
            
        else:
            print_grid()

if __name__ == "__main__":
    # Quick Test
    game = Connect4()
    game.drop_token(3, Connect4.RED) # Col 3 (index) -> "4"
    game.drop_token(3, Connect4.YELLOW)
    game.render_ascii(animate_drop=(2, 3))
