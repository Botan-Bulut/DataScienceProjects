"""
Coding in utf-8
2024-05-10
Botan BULUT
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
import sys
import pdb


def script_help() -> None:
    """Prints the script help file"""

    help_string ="""
*** Connect the dots AI solver script ***

--help: Displays the help.

Usage: main.py <path to problem file.>
"""
    
    print(help_string)


def validate_user_input(args: list) -> None:
    """Validate command line arguments from the user."""
   
    flattened_arguments = ' '.join(args)
    
    # If user asked for help we print help string:
    if '--help' in flattened_arguments:
        script_help()
        sys.exit(0)

    # File path and script name
    if len(args) != 2:
        print('To get help. Type "main.py --help".')
        print("Script terminated with error.")
        sys.exit(1)

    # Validate if the problem path exists.
    path = sys.argv[1]
    if not os.path.exists(path):
        print(f'File not found: {path}')
        print("Script terminated with error.")
        sys.exit(1)


class Game():
    """Main game class."""
    

    def __init__(self, problem_path: str) -> None:
        
        # Obtain variables:
        self.problem_path = problem_path
        self.game = None  # This is NumPy array representation of the game.
        self.start_loc = None
        self.solved_status = False  # Update this when AI solves the puzzle.
        self.solution_dict = None
        self.solution_path = None  # Solution path of the puzzle.
        self.unfilled_locations = None  # Unfilled row index locations.
        return None
    

    def read_game(self) -> None:
        """This method reads game file and converts it to NumPy array."""
        
        # Define lines:
        lines = []
        
        # Read file:
        with open(self.problem_path, 'r') as f:
            for line in f:
                lines.append(list(line[:-1]))
        
        # Modify game array function with padding:
        temp = np.array(lines, dtype='int')
        self.game = np.zeros(shape=(temp.shape[0] 
            + 2, temp.shape[1] + 2))
        
        # Copy inner array to padded array:
        self.game[1:-1, 1:-1] = temp

        # Get the start location:
        start_loc = np.where(self.game == 2)
        self.start_loc = (start_loc[0][0], start_loc[1][0])
        
        # Get unfilled locations:
        self.unfilled_locations = np.where(self.game == 1)[0].tolist()
        
        # Append the row index of initial location:
        self.unfilled_locations.append(self.start_loc[0])
        self.unfilled_locations = sorted(self.unfilled_locations,
                reverse=False)

        return None
    

    def plot_game(self) -> None:
        """This method is used for plotting the game."""
        
        cmap = ListedColormap(['Black', 'Gray', 'Red'])
        fig = plt.figure('Game Plot', figsize=(5, 5), dpi=150)
        ax = fig.add_subplot()
        ax.set_title('Game Plot', fontsize=16, fontweight='bold')
        ax.imshow(self.game,
                cmap=cmap,
                aspect='equal',
                extent=(0, self.game.shape[1],
                    0, self.game.shape[0]),
                origin='lower')

        ax.grid(True, linewidth=2, color='k')
        
        ax.tick_params(labelbottom=False,
                labelleft=False)

        ax.set_xticks(np.arange(0, self.game.shape[1] + 1, 1))
        ax.set_yticks(np.arange(0, self.game.shape[0] + 1, 1))
        
        # If puzzle is solved:
        if self.solved_status == True:
        
            y = np.array([y[0] for y in self.solution_dict.keys()])
            x = np.array([x[1] for x in self.solution_dict.keys()])
            y = y + 0.5
            x = x + 0.5

            # Plot solution path:
            ax.plot(x,
                    y,
                    ls='--',
                    color='yellow',
                    lw=3,
                    marker='o',
                    markersize=10,
                    mec='k')

        ax.invert_yaxis()
        fig.tight_layout()
        plt.show()
        return None
    

    def solver(self, target_loc: tuple,
            updated_game: np.ndarray,
            solution_dict: dict) -> None:
        """
        AI solver function: implements the search algorihms recursively.
        """
        
        # Return and set the status if win:
        if self.solver_check_win_status(current_game=updated_game):
            print('** AI successfully found a solution! **')
            self.solved_status = True
            self.solution_path = list(solution_dict.values()).copy()
            self.solution_dict = solution_dict.copy()

            # Call the plot once again:
            self.plot_game()
            return None
        
        available_locations = (self
                .solver_get_available_moves(current_loc=target_loc,
                    current_game=updated_game))
        
        # If there is no possible moves return none and inform user:
        if len(available_locations) == 0:
            return None
        
        # Else recursively call path expansion for all locations:
        else:
            for location in available_locations:
                
                # Get direction:
                direction = self.solver_get_move_type(initial_loc=target_loc,
                        moved_loc=location)
                
                # Add location and direction pair to dict:
                updated_dict = solution_dict.copy()
                updated_dict[location] = direction
                post_move_game = (self
                        .solver_get_updated_game(current_game=updated_game,
                            move=location))
                
                # Recurse:
                self.solver(target_loc=location,
                        updated_game=post_move_game,
                        solution_dict=updated_dict)
    

    def solver_get_updated_game(self,
            current_game: np.ndarray,
            move: tuple) -> np.ndarray:
        """This method returns updated game array."""

        # Get the update:
        updated_game = current_game.copy()
        updated_game[move] = 2
        return updated_game
    

    def solver_get_move_type(self,
            initial_loc: tuple,
            moved_loc: tuple) -> str:
        """
        This function determines finds the direction associated with the move.
        """
        
        # Calculate the result
        result = np.array(moved_loc) - np.array(initial_loc)

        # Conditionally assign the result:
        if np.array_equal(result, np.array((1, 0))):
            return 'down'
        elif np.array_equal(result, np.array((-1, 0))):
            return 'up'
        elif np.array_equal(result, np.array((0, 1))):
            return 'right'
        else:
            return 'left'
    

    def solver_get_available_moves(self,
            current_loc: tuple,
            current_game: np.ndarray) -> list:
        """
        This function gets the available moves and returns then as a list.
        """
        
        # Generate move locations:
        possible_move_locations = [
                (current_loc[0] + 1, current_loc[1]),
                (current_loc[0] - 1, current_loc[1]),
                (current_loc[0], current_loc[1] + 1),
                (current_loc[0], current_loc[1] - 1)]

        # Evaluate available locations via filter:
        filter_func = lambda loc: True if current_game[loc] == 1 else False
        available_locations = list(filter(filter_func,
            possible_move_locations))
        
        return available_locations


    def solver_check_win_status(self, current_game: np.ndarray) -> bool:
        """
        This method checks the game end status from using the provided
        game array.
        """
        current_fills = (
                sorted(np.where(current_game == 2)[0].tolist(),
                    reverse=False))
        
       # If all the empty spaces is filled:
        if current_fills == self.unfilled_locations:
            return True  # returns true when all white is filled.
        else:
            return False


def main() -> None:
    """Main function of the script"""
    
    # Validate command line arguments:
    validate_user_input(args=sys.argv)

    # Generate game:
    game = Game(problem_path=sys.argv[1])
    game.read_game()
    game.plot_game()
    game.solver(target_loc=game.start_loc,
            updated_game=game.game,
            solution_dict={game.start_loc: 'start'})
    
    # Print game solution
    print('-'.join(game.solution_path))
    return None

if __name__=='__main__':
    main()
else:
    print('The script has been imported succefully.')
