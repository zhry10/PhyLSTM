# Physics-informed LSTM Network（PhyLSTM)
We introduce an innovative physics-informed LSTM framework for metamodeling of nonlinear structural systems with scarce data. The basic concept is to incorporate available, yet incomplete, physics knowledge (e.g., laws of physics, scientific principles) into deep long short-term memory (LSTM) networks, which constrains and boosts the learning within a feasible solution space. The physics constraints are embedded in the loss function to enforce the model training which can accurately capture latent system nonlinearity even with very limited available training datasets. Specifically for dynamic structures, physical laws of equation of motion, state dependency and hysteretic constitutive relationship are considered to construct the physics loss. The embedded physics can alleviate overfitting issues, reduce the need of big training datasets, and improve the robustness of the trained model for more reliable prediction with extrapolation ability. As a result, the physics-informed deep learning paradigm outperforms classical non-physics-guided data-driven neural networks.


For more information, please refer to the following:
* Zhang, R., Liu, Y., & Sun, H. (2020). [Physics-informed multi-LSTM networks for metamodeling of nonlinear structures](https://www.sciencedirect.com/science/article/pii/S0045782520304114). Computer Methods in Applied Mechanics and Engineering 369, 113226.


## Citation
<pre>
@article{zhang2020physics,
  title={Physics-informed multi-LSTM networks for metamodeling of nonlinear structures},
  author={Zhang, Ruiyang and Liu, Yang and Sun, Hao},
  journal={Computer Methods in Applied Mechanics and Engineering},
  volume={369},
  pages={113226},
  year={2020},
  publisher={Elsevier}
}
</pre>
