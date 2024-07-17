# Near Neighbor

> Findout neighbor infomation

<div align=center><img src="./transform_route.png" width=600></div>

## Neighbor Aggregation

- 3d

| cities | transform |
| :----: | :-------: |
| ![alt](./images_3d/3d_cities/output.gif) | ![alt](./images_3d/3d_transform/output.gif) |

- 2d

| class | images_att | images_cos | images_euc |
| :---: | :--------: | :--------: | :--------: |
| cities | ![alt](./images_att/cities/output.gif) | ![alt](./images_cos/cities/output.gif) | ![alt](./images_euc/cities/output.gif) |
| transform | ![alt](./images_att/transform/output.gif) | ![alt](./images_cos/transform/output.gif) | ![alt](./images_euc/transform/output.gif) |

## TSP

- different method to get solution `route:List[int]` of TSP problem

| Greedy(Nearest Neighbor) | Simulated Annealing | Genetic Algorithm |
| :----------------------: | :-----------------: | :---------------: |
| ![alt](./images_tsp/route/NEAREST_NEIGHBOR.png) | ![alt](./images_tsp/route/SIMULATED_ANNEALING.png) | ![alt](./images_tsp/route/GENETIC_ALGORITHM.png) |