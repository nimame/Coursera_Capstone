FROM jupyter/datascience-notebook:1386e2046833

# RUN conda install -c conda-forge geopandas rtree pyproj shapely fiona
COPY requirements.txt /tmp/
RUN conda install -c conda-forge --yes --file /tmp/requirements.txt && \
    pip install --requirement /tmp/requirements.txt && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER

