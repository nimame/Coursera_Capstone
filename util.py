# System
import os
import datetime
import re
import json
import pickle

# Basics
import numpy as np
import pandas as pd

# Plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns

# GIS
from owslib.wfs import WebFeatureService
from owslib.wms import WebMapService
from shapely.geometry import MultiPolygon, Polygon
import geopandas as gpd

# ML
from sklearn.preprocessing import StandardScaler

# Tools
from tqdm import tqdm_notebook as tqdm
import adjustText as aT
import foursquare
from googletrans import Translator

HAMBURG_GEOSERVICES_URL = "https://geodienste.hamburg.de"
HAMBURG_ADMINBORDERS_URL = f"{HAMBURG_GEOSERVICES_URL}/HH_WFS_Verwaltungsgrenzen?REQUEST=GetCapabilities&SERVICE=WFS"
HAMBURG_POPULATION_URL = f"{HAMBURG_GEOSERVICES_URL}/HH_WMS_Statistische_Gebiete?SERVICE=WMS&REQUEST=GetCapabilities"

GEOGRAPHY_NAME = "name"
ADMIN_LEVEL_NAME = "admin_level"
VARNAME_JSON_FILE = 'data/hamburg_socioecon_varnames.json'
SOCIOECON_PKL_FILE = "data/hamburg_socioecon.pkl"
HAMBURG_SOCIO_ECONOMICS_2017 = "https://www.statistik-nord.de/fileadmin/Dokumente/Datenbanken_und_Karten/Stadtteilprofile/StadtteilprofileBerichtsjahr2017.xlsx"
HAMBURG_SOCIO_ECONOMICS_2013_2016 = "https://www.statistik-nord.de/fileadmin/Dokumente/Datenbanken_und_Karten/Stadtteilprofile/Stadtteilprofile-Berichtsjahre-2013-2016.xlsx"
CATEGORY_SELECTION = {
    "Groceries": {"Supermarket": "52f2ab2ebcbc57f1066b8b46", "Grocery Store": "4bf58dd8d48988d118951735",
                  "Organic Grocery": "52f2ab2ebcbc57f1066b8b45"},
    "Education": {"High School": "4bf58dd8d48988d13d941735", "Middle School": "4f4533814b9074f6e4fb0106",
                  "Child Care Service": "5744ccdfe4b0c0459246b4c7", "Playground": "4bf58dd8d48988d1e7941735"},
    "Health": {"Physical Therapist": "5744ccdfe4b0c0459246b4af",
               "Doctor's Office": "4bf58dd8d48988d177941735"},
    "Public Transportation": {"Bus Stop": "52f2ab2ebcbc57f1066b8b4f",
                              "Light Rail Station": "4bf58dd8d48988d1fc931735",
                              "Metro Station": "4bf58dd8d48988d1fd931735",
                              "Train Station": "4bf58dd8d48988d129951735"},
    "Gastronomy": {"Food": "4d4b7105d754a06374d81259"},
    "Nightlife": {"Nightlife Spot": "4d4b7105d754a06376d81259"}}

FOURSQUARE_API_URL = "https://api.foursquare.com/v2"
FOURSQUARE_CATEGORIES_ENDPOINT = f"{FOURSQUARE_API_URL}/venues/categories"
FOURSQUARE_SEARCH_ENDPOINT = f"{FOURSQUARE_API_URL}/venues/search"
FOURSQUARE_CATEGORIES_FILE = 'data/foursquare_categories.pkl'
FOURSQUARE_ZOOM_LOG_FILE = "out/foursquare_zoom_log.pkl"
FOURSQUARE_LOG_FILE = "out/foursquare_venues_log.pkl"
FOURSQUARE_DATA_FILE = "data/foursquare_data.pkl"


def get_color_range(n):
    fig, ax = plt.subplots(figsize=(10, 1))

    for i in range(0, n):
        ax.hlines(0, 10/n*i-10/n, 10/n*i,
                  color=f'C{i}', linewidth=18)
    ax.set_axis_off()


def get_triangular_bool(n_cols):
    triang = np.repeat(False, n_cols ** 2).reshape(n_cols, n_cols)
    triang[np.triu_indices_from(triang)] = True
    return triang


def plot_heatmap(data, columns):
    fig, ax = plt.subplots(figsize=(15, 7))
    numbering = range(1, len(columns) + 1)
    sns.heatmap(
        data[columns].corr().round(2),
        cmap='coolwarm',
        annot=True, linewidths=.1,
        mask=get_triangular_bool(len(columns)),
        yticklabels=[f"{i}. {columns[i - 1]}" for i in numbering],
        xticklabels=numbering,
        ax=ax
    )


def scale_df(df):
    return pd.DataFrame(StandardScaler().fit_transform(df), columns=df.columns,
                        index=df.index)


def get_service_url(name, request="GetCapabilities", service="WFS"):
    return f"{HAMBURG_GEOSERVICES_URL}/{name}?REQUEST={request}&SERVICE={service}"


def get_service(name, service, version="2.0.0"):
    # https://geodienste.hamburg.de/HH_WFS_Bebauungsplaene?REQUEST=GetCapabilities&SERVICE=WFS
    # https://geodienste.hamburg.de/HH_WFS_Mietenspiegel?REQUEST=GetCapabilities&SERVICE=WFS
    # https://geodienste.hamburg.de/HH_WFS_Bebauungsplaene?REQUEST=GetCapabilities&SERVICE=WFS
    # https://geodienste.hamburg.de/HH_WFS_Bodenrichtwerte?REQUEST=GetCapabilities&SERVICE=WFS
    # https://geodienste.hamburg.de/HH_WFS_Bodenschaetzung?REQUEST=GetCapabilities&SERVICE=WFS
    # https://geodienste.hamburg.de/HH_WFS_Statistik_Stadtteile_Bevoelkerung?REQUEST=GetCapabilities&SERVICE=WFS
    if service == "WFS":
        ws = WebFeatureService(url=get_service_url(name, service=service), version=version)
    elif service == "WMS":
        ws = WebMapService(url=get_service_url(name, service=service), version=version)
    return ws


def list_service_contents(name, service="WFS", version="2.0.0"):
    ws = get_service(name=name, service=service, version=version)
    print(f"contents: {', '.join(list(ws.contents))}")


def get_geodata(name, service, type_name, request='GetFeature', version="2.0.0"):
    ws = get_service(name=service, service="WFS", version="2.0.0")
    # url = get_service_url(name, service=service)
    params = dict(service=service, version=version, request=request,
                  typeName=type_name, outputFormat='application/geo+json')

    r = ws.getfeature(typename=type_name, outputFormat="application/geo+json")
    return r.read()


def import_socioecon_data(paths=None):
    if not isinstance(paths, list) and paths is None:
        paths = [HAMBURG_SOCIO_ECONOMICS_2017, HAMBURG_SOCIO_ECONOMICS_2013_2016]

    meta = dict()
    data_collection = []

    for path in paths:

        for sheet in pd.ExcelFile(path).sheet_names:

            # Download the headers of the files which include variable names and descriptions
            header = pd.read_excel(path, header=None, nrows=4, sheet_name=sheet)

            translator = Translator()

            # If run the second time use the JSON file that contains the meta data,
            # including variable names and descriptions
            if os.path.isfile(VARNAME_JSON_FILE):
                with open(VARNAME_JSON_FILE, 'r') as f:
                    meta = json.loads(f.read())

            header.loc[2, 0] = header.loc[1, 0]
            updated = False

            # Translate variable names and descriptions to English
            for _, content in header.loc[2:3, :].iteritems():

                label = content[2]
                description = content[3]

                if isinstance(label, str):

                    split = re.split("(\\(.*$)", label)

                    if label not in meta.keys():
                        meta[label] = {}

                    if isinstance(description, str):
                        description = description.strip()

                        if "description" not in meta[label]:
                            updated = True
                            meta[label]["description"] = description

                        if "description_en" not in meta[label]:
                            updated = True
                            meta[label]["description_en"] = translator.translate(description, src="de", dest="en").text
                            meta[label]["machine_translated"] = True

                        if "label_en" not in meta[label]:
                            updated = True
                            label_en = split[0].strip()
                            meta[label]["label_en"] = translator.translate(label_en, src="de", dest="en").text
                            meta[label]["machine_translated"] = True

                        if "variable_name" not in meta[label]:
                            updated = True
                            varname = meta[label]["label_en"].lower().replace(" ", "_").replace("%", "_pct")
                            varname = re.sub(r"[^A-Za-z0-9_]", "", varname)
                            varname = re.sub(r"_+", "_", varname)
                            meta[label]["variable_name"] = varname
                            meta[label]["machine_translated"] = True

            if updated:
                with open(VARNAME_JSON_FILE, 'w') as outfile:
                    json.dump(meta, outfile, sort_keys=True, indent=4, ensure_ascii=False)

            rename_map = dict()

            for key in meta.keys():
                rename_map[key] = meta[key]['variable_name']

            # Download the data without the headers
            data = pd.read_excel(path, header=None, skiprows=4, sheet_name=sheet)

            # Set the column names to the original German variable names
            data.columns = header.loc[2, :]

            # Rename the column names using the meta data (potentially updated manually)
            data = data.rename(columns=rename_map)

            # Extract the year
            year = re.findall("([0-9]+)", sheet)[0]
            data["year"] = year
            data_collection.append(data)

    # Concatenate the the yearly data into one DataFrame
    data = pd.concat(data_collection, ignore_index=True, sort=False)
    # Sort columns
    data = data[[GEOGRAPHY_NAME] + sorted(data.columns.drop(GEOGRAPHY_NAME))]
    # Specify the admin levels
    data[ADMIN_LEVEL_NAME] = 10
    data.loc[data[GEOGRAPHY_NAME].isin([
        "Hamburg",
        "Wandsbek",
        "Hamburg-Nord",
        "Hamburg-Mitte",
        "Altona",
        "Bergedorf",
        "Harburg"
    ]) & (data.area_in_km2 > 40), ADMIN_LEVEL_NAME] = 9
    data.loc[data[GEOGRAPHY_NAME] == "Hamburg", ADMIN_LEVEL_NAME] = 8

    # Some columns contain non-numeric characters as missing value indicators, convert those columns to numeric with nan
    non_numeric_columns = [GEOGRAPHY_NAME, 'year']
    data.loc[:, ~data.columns.isin(non_numeric_columns)] = data.drop(columns=non_numeric_columns).apply(pd.to_numeric,
                                                                                                        errors='coerce')
    data.to_pickle(SOCIOECON_PKL_FILE)

    return data


def get_foursquare_categories():
    client = foursquare.Foursquare(client_id=os.getenv('FOURSQUARE_CLIENT_ID'),
                                   client_secret=os.getenv('FOURSQUARE_SECRET'), version='20191001')
    params = dict(
        client_id=os.getenv('FOURSQUARE_CLIENT_ID'),
        client_secret=os.getenv('FOURSQUARE_SECRET'),
        includeSupportedCC="true"
    )
    response = client.venues.categories(params=params)
    with open(FOURSQUARE_CATEGORIES_FILE, 'wb') as f:
        pickle.dump(response, f)


def find_category(categories, category_id, path=[]):
    for c in categories:
        category = c.copy()
        subcategories = None
        if "categories" in category.keys():
            subcategories = category["categories"]
            del category["categories"]
        if category['id'] == category_id:
            return category, path
        elif subcategories:
            p = path.copy()
            p.append(category)
            c, p = find_category(subcategories, category_id, p)
            if c:
                return c, p
    return None, None


def create_raster(shape, n: int = 4500):
    bounds = shape.bounds
    x0 = bounds[0]
    y0 = bounds[1]
    x1 = bounds[2]
    y1 = bounds[3]
    a = (x1 - x0) / np.sqrt(n)
    polys = []
    for x in np.arange(x0, x1, a):
        for y in np.arange(y0, y1, a):
            poly = Polygon([(x, y), (x, y + a), (x + a, y + a), (x + a, y), (x, y)])
            if poly.intersects(shape):
                polys.append(poly)
    return gpd.GeoDataFrame(geometry=gpd.GeoSeries(polys))


def get_foursquare_data(locations: gpd.GeoDataFrame, topics: list, layer: int = 0):
    client = foursquare.Foursquare(client_id=os.getenv('FOURSQUARE_CLIENT_ID'),
                                   client_secret=os.getenv('FOURSQUARE_SECRET'), version='20191001')

    category_selection = {key: CATEGORY_SELECTION[key] for key in topics}

    log = locations.iloc[[0]]
    for topic, selection in category_selection.items():
        locations['categories'] = ",".join(selection.values())
        locations['topic'] = topic
        log = log.append(locations, ignore_index=True, sort=False)
    log = log.drop(0).reset_index(drop=True)

    log['count'] = None
    log['time'] = None
    log['layer'] = None

    if os.path.isfile(FOURSQUARE_LOG_FILE):
        with open(FOURSQUARE_LOG_FILE, 'rb') as f:
            log = pickle.load(f).append(log, ignore_index=True, sort=False).drop_duplicates(["categories", "geometry"])

    zoom = {}

    if os.path.isfile(FOURSQUARE_ZOOM_LOG_FILE):
        with open(FOURSQUARE_ZOOM_LOG_FILE, 'rb') as f:
            zoom = pickle.load(f)

    if f"layer{layer}" not in zoom.keys():
        zoom[f"layer{layer}"] = []

    result = dict()
    if os.path.isfile(FOURSQUARE_DATA_FILE):
        with open(FOURSQUARE_DATA_FILE, 'rb') as f:
            result = pickle.load(f)

    mask = log['count'].isna()
    counter = 0

    for i, loc in tqdm(log[mask].iterrows(), total=sum(mask), desc=f"Areas (layer: {layer})"):
        counter += 1
        bounds = loc.geometry.bounds

        params = dict(
            client_id=os.getenv('FOURSQUARE_CLIENT_ID'),
            client_secret=os.getenv('FOURSQUARE_SECRET'),
            sw=",".join(map(str, bounds[1:None:-1])),
            ne=",".join(map(str, bounds[4:1:-1])),
            intent="browse",
            categoryId=loc.categories,
            limit=50
        )

        response = client.venues.search(params=params)

        for venue in response['venues']:
            result[venue['id']] = venue
            result[venue['id']]['topic'] = loc.topic

        log.loc[i, "count"] = len(response['venues'])
        log.loc[i, "time"] = datetime.datetime.now()
        log.loc[i, "layer"] = layer

        if len(response['venues']) >= 50:
            print(f"added geometry to zoom, size: {bounds[2] - bounds[0]}")
            zoom[f"layer{layer}"].append(loc[["geometry", "topic"]])

        if (counter % 25 == 0) or counter >= sum(mask):
            with open(FOURSQUARE_DATA_FILE, 'wb') as f:
                pickle.dump(result, f)

            with open(FOURSQUARE_ZOOM_LOG_FILE, 'wb') as f:
                pickle.dump(zoom, f)

            with open(FOURSQUARE_LOG_FILE, 'wb') as f:
                pickle.dump(log, f)

    for el in zoom[f"layer{layer}"]:
        result = get_foursquare_data(create_raster(el.geometry, 9), [el.topic], layer=layer + 1)
    del zoom[f"layer{layer}"]

    return result


def get_cluster_summary(data: pd.DataFrame, cluster_mask: pd.Series, y: str, n=5):
    total_means = data.mean()
    data_scaled = scale_df(data)
    total_means_scaled = data_scaled.mean()
    mean = data[cluster_mask].mean().rename("Cluster Mean")
    mean_diff = data[cluster_mask].mean().subtract(total_means).rename("Diff. Grand Mean")
    scaled_diff = "Scaled Diff."
    mean_diff_scaled = data_scaled[cluster_mask].mean().subtract(total_means_scaled).rename(scaled_diff)
    summary_df = pd.concat([mean, mean_diff, mean_diff_scaled], axis=1)

    return summary_df.loc[[y], :].append(summary_df.sort_values(scaled_diff,
                                                                ascending=False).iloc[
                                             pd.np.r_[0:n, -n:0]]).drop_duplicates()


def plot_variable_on_map(data: pd.DataFrame, var: str, label=None, title=None, quantiles=False, discrete=False,
                         adjust_texts=False):

    if label is None:
        label = var

    fig, ax = plt.subplots(figsize=(24, 18))

    if quantiles:
        data.plot(ax=ax, edgecolor='white', cmap='PuRd', column=var, legend=True, scheme='QUANTILES')

    elif discrete:
        n_classes = len(data[label].unique())
        cmap = mpl.cm.get_cmap('tab20c')
        norm = mpl.colors.Normalize(vmin=-1, vmax=n_classes + 1)
        # ListedColormap([f'C{i}' for i in range(0, n_classes)])
        data.plot(ax=ax, edgecolor='white', column=var, legend=True,
                  cmap=ListedColormap([cmap(norm(i)) for i in range(0, n_classes)]),
                  alpha=1,
                  legend_kwds={'prop': {'size': 16}})

    else:
        data.plot(ax=ax, edgecolor='white', cmap='PuRd', column=var, legend=True,
                  legend_kwds={'orientation': "horizontal",
                               'pad': 0.05, 'shrink': .7, 'aspect': 25})
        # ax.get_legend().set_fontsize(12)
        ax.figure.axes[1].tick_params(labelsize=14)
        ax.figure.axes[1].set_xlabel(label, fontsize=16)

    texts = []
    if 'representative_point' not in data.columns:
        data['representative_point'] = data.representative_point()

    for i, row in data.set_geometry(data['representative_point']).iterrows():
        texts.append(ax.annotate(s=i, xy=(row.geometry.xy[0][0], row.geometry.xy[1][0]), alpha=0.8,
                                 ha='center'))

    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=20)
    plt.axis('equal')
    if adjust_texts:
        aT.adjust_text(texts)
