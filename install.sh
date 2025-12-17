# Create a virtual environment
uv venv --python 3.11 
source .venv/bin/activate  # Windows: venv\Scripts\activate

cd mujoco_playground

# Install CUDA 12 jax
uv pip install -U "jax[cuda12]"

#Verify GPU backend: python -c "import jax; print(jax.default_backend())" should print gpu

uv pip install -e ".[all]"

cd .. 