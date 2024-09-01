---
title: 'Graph Search'
date: 2024-08-27
permalink: /posts/2024/08/graph-search/
tags:
  - cs
---

One of my (sub) goals was to study Topological sort while studying autograd internals. I realized I could not do justice
studying it in isolation. And while I would have to study depth-first search (which topological sort is based on) from scratch, might as well add BFS, Dijkstra and A* to the cart.


### Outline
- [Toy problem: Maze solver](#toy-problem-maze-solver)
- [DFS](#depth-first-search)
- [BFS](#breadth-first-search)
- Dijkstra
- A*
- Topological sort


```python
import random

import matplotlib.pyplot as plt
import numpy as np
from IPython import display
```

### Toy Problem: Maze solver
A maze is a mesh of cells with distinct paths that can (or cannot) be traversed, it is a graph. We will pick a toy problem, with the goal of finding a path from the origin (top-left) to 
the sink (bottom-right) of a randomly generated maze/graph. There are [several](https://en.wikipedia.org/wiki/Graph_(abstract_data_type)#Common_data_structures_for_graph_representation) ways to encode a graph. Our graph will be encoded as an adjacency matrix, with `1` representing a wall and `0` representing a pathway.

I will use a simple recusive backtracking algorithm to generate mazes. [Here](https://weblog.jamisbuck.org/2011/2/7/maze-generation-algorithm-recap) is a goldmine of other algorithms if you are interested.


```python
def draw(maze: np.ndarray):
  """Auto-updating plot"""
  display.clear_output(wait=True)
  plt.figure(figsize=(4, 4))
  plt.axis("off")
  plt.imshow(maze, cmap="viridis_r")
  plt.show()

def generate_maze(n: int) -> np.ndarray:
  """Generates a random N*N maze using recursive backtracking."""
  # Make `n` odd. Why?
  n -= (n % 2)
  n += 1
  maze = np.ones((n, n), dtype=np.int32)

  # Opening at the top and bottom. We choose these points
  # because we can guarantee that an odd-maze will not have doubly-thick walls.
  maze[0][1] = maze[-1][-2] = 0

  # Direction vectors. Moving by 2 units ensures
  # that we skip over the walls and move from one potential passage to next.
  directions = [(0, 2), (0, -2), (2, 0), (-2, 0)]

  # Choose a random odd coordinate.
  start = (random.randrange(1, n, 2), random.randrange(1, n, 2))
  maze[start] = 0
  stack = [start]

  while stack:
    cy, cx = stack[-1]

    # Get neighbors in a random order.
    random.shuffle(directions)
    found_unvisited_neighbor = False

    for dx, dy in directions:
      nx, ny = cx + dx, cy + dy

      # Check if the candidate cell is not out of bounds and is a wall.
      if (0 <= nx < n and 0 <= ny < n and maze[ny][nx] == 1):
        # Pave thru the wall.
        maze[ny][nx] = 0
        maze[cy + dy // 2][cx + dx // 2] = 0
        stack.append((ny, nx))
        found_unvisited_neighbor = True
        break
    
    # Backtrack if all neighbors have been visited.
    if not found_unvisited_neighbor:
      stack.pop()
    
  return maze

# random.seed(42)
maze = generate_maze(30)
draw(maze)

```


    
![png](output_3_0.png)
    


Our goal now, is to search for a path from the origin (top-left) to sink (bottom-right). 

### Depth-First Search

A key feature of this search is that it exhaustively searches through all possible sub-vertices connected to a given vertex, before backtracking and moving to a different vertex at the same level. DFS has no foresight of how far it is from its goal. For example, if DFS is _very_ close to a solution, and then yeets off to a random sub-vertex, it will not come back until it has exhaustively searched in that wrong direction.


```python
def dfs(maze: np.ndarray,
        start: tuple[int, int] = (0, 1),
        end: tuple[int, int] = (-1, -2),
        visualize: bool = False) -> bool:
  """Checks if a path exists between `start` and `end`."""
  assert maze[start] == maze[end] == 0, "One of `start` or `end` is a wall."
  rows, cols = maze.shape

  # Create a matrix of visited nodes.
  visited = np.zeros_like(maze).astype(np.bool_)
  visited[start] = True

  # Define search directions.
  directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

  candidates = [start]
  found = False
  while candidates:

    if visualize:
      # Color visited nodes.
      draw(np.where(visited, 0.5, maze))

    # If we hit the end, a path exists.
    if visited[end]:
      found = True
      break

    # We pick the most recent candidates to search forward (LIFO).
    # This is where "Depth" in DFS comes from.
    cy, cx = candidates.pop(-1)

    for dy, dx in directions:
      ny, nx = cy + dy, cx + dx
      if (0 <= nx < cols and 0 <= ny < rows and not maze[ny][nx] == 1 and
          not visited[ny][nx]):
        candidates.append((ny, nx))
        visited[ny][nx] = True

  return found

dfs(maze, visualize=True)
```


    
![png](output_5_0.png)
    





    True



DFS search time is proportional to the number of vertices in our adjacency matrix - $O(|N|^2)$. The space required to store intermediate states is proportional to the number of vertices (since `visited` array and `candidates` stack are the only auxiliary objects in our function), therefore $O(|N|^2)$.

### Breadth-First Search

A natural modification to DFS could be made on candidate selection. In fact, all the search algorithms we will see today are merely intelligent ways of 'choosing where to search'.

Instead of going down the rabbit-hole on a single vertex, what if we first explore all vertices available at a given level?


```python
def bfs(maze: np.ndarray,
        start: tuple[int, int] = (0, 1),
        end: tuple[int, int] = (-1, -2),
        visualize: bool = False) -> bool:
  """Checks if a path exists between `start` and `end`."""
  assert maze[start] == maze[end] == 0, "One of `start` or `end` is a wall."
  rows, cols = maze.shape

  # Create a matrix of visited nodes.
  visited = np.zeros_like(maze).astype(np.bool_)
  visited[start] = True

  # Define search directions.
  directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

  candidates = [start]
  found = False
  while candidates:

    if visualize:
      # Color visited nodes.
      draw(np.where(visited, 0.5, maze))

    # If we hit the end, a path exists.
    if visited[end]:
      found = True
      break

    # We pick candidates in the order they were added, like a queue (FIFO).
    # It gives a feeling of parallel search at a given level.
    cy, cx = candidates.pop(0)

    for dy, dx in directions:
      ny, nx = cy + dy, cx + dx
      if (0 <= nx < cols and 0 <= ny < rows and not maze[ny][nx] == 1 and
          not visited[ny][nx]):
        candidates.append((ny, nx))
        visited[ny][nx] = True

  return found

bfs(maze, visualize=True)
```


    
![png](output_8_0.png)
    





    True



BFS has the same time and space complexity as DFS in this example - $O(|N|^2)$.

> Note: We can do a microbenchmark to see that neither of DFS or BFS is relatively faster. This is an expected outcome for large, uniformly random mazes. Real world graphs typically carry structure bias towards depth (or width). Hence, choice of DFS/BFS should be informed by the structure of the graph.


```python
# takes ~1 min
%timeit -n 2 -r 10 dfs(generate_maze(500))
%timeit -n 2 -r 10 bfs(generate_maze(500))
```

    1.29 s ± 128 ms per loop (mean ± std. dev. of 10 runs, 2 loops each)
    1.28 s ± 147 ms per loop (mean ± std. dev. of 10 runs, 2 loops each)


### Guided Search

In certain cases we can 'tell' how far we are from the goal. This information can be used to speed up search (and find the shortest path), forming the basis of Dijkstra's algorithm.


```python
import heapq

def dist(src: tuple[int, int], dest: tuple[int, int]):
  return abs(src[0] - dest[0]) + abs(src[1] - dest[1])

def dijkstra(maze: np.ndarray,
             start: tuple[int, int] = (0, 1),
             end: tuple[int, int] = (-1, -2),
             visualize: bool = False) -> bool:
  """Checks if a path exists between `start` and `end`."""
  assert maze[start] == maze[end] == 0, "One of `start` or `end` is a wall."
  rows, cols = maze.shape

  # Create a matrix of visited nodes.
  visited = np.zeros_like(maze).astype(np.bool_)
  visited[start] = True

  # Define search directions.
  directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

  candidates = [(start, dist(start, end))]
  found = False
  while candidates:
    print(candidates)
    break
    if visualize:
      # Color visited nodes.
      draw(np.where(visited, 0.5, maze))

    # If we hit the end, a path exists.
    if visited[end]:
      found = True
      break

    # Pick the candidate closest to our goal.
    # cy, cx = ??

    for dy, dx in directions:
      ny, nx = cy + dy, cx + dx
      if (0 <= nx < cols and 0 <= ny < rows and not maze[ny][nx] == 1 and
          not visited[ny][nx]):
        candidates.append((ny, nx))
        visited[ny][nx] = True

  return found

dijkstra(maze, visualize=True)
```

    [((0, 1), 4)]





    False



Ok, back to `toposort`. All autograd engines construct a directed computational graph of a given function. This 
function takes a set of input tensor(s), and through layers of transformations, evaluates to a scalar `loss` value. 
The goal is often to minimize this `loss` by moving along the derivative. But in a graph with (easily) hundreds of nodes and edges, an autograd engine must know the exact order of reverse traversal.

In summary, we have a root node which is the `loss`, and we need a list of all dependencies to `loss` in the _order_
they impact it. This is searching along the depth from the `loss` node: depth-first search.


```python

```
