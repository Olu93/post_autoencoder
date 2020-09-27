from pathlib import Path
import pandas as pd
import numpy as np
import glob
from pprint import pprint
from plyfile import PlyData
import pyvista as pv
import os
import matplotlib.pyplot as plt
from collections import Counter
from itertools import chain
import inspect
import io
from helper.config import DEBUG, DATA_PATH, CLASS_FILE
from tqdm import tqdm


class DataSet:
    data_descriptors = []
    full_data = []
    stats_path = None
    schemes = ["**/*.off", "**/*.ply"] if not DEBUG else ["**/*.ply"]
    has_descriptors = None
    has_stats = False
    has_loaded_data = False
    has_poly_data = False
    has_outliers = False
    class_member_ships = {}

    def __init__(self, search_paths, stats_path=None):
        data_folder = [Path(path) for path in search_paths]
        self.stats_path = stats_path
        self.data_file_paths = list(chain(*[glob.glob(str(path), recursive=True) for path in data_folder]))

    def read(self):
        raise NotImplementedError

    def load_files_in_memory(self):
        assert self.has_descriptors, f"Dunno the file locations. Run {self.read.__name__} function first."
        len_of_ds = len(self.data_descriptors)
        print(f"Loading {len_of_ds} models into memory!")
        full_data_generator = ({
            "meta_data": file,
            "data": self._load_ply(file["path"]) if not file["type"] == ".off" else self._load_off(file["path"])
        } for file in tqdm(self.data_descriptors, total=len_of_ds))
        self.full_data = list(full_data_generator)
        self.has_loaded_data = True
        print(f"Finished {inspect.currentframe().f_code.co_name}")

    def compute_shape_statistics(self):
        assert self.has_poly_data, f"No pyvista objects available. Run {self.convert_all_to_polydata.__name__} first"
        self.full_data = [dict(object_descriptor, statistics=self._compute_statistics(object_descriptor)) for object_descriptor in self.full_data]
        self.all_statistics = pd.DataFrame([mesh_object["statistics"] for mesh_object in self.full_data])
        self.has_stats = True
        print(f"Finished {inspect.currentframe().f_code.co_name}")

    def save_statistics(self, stats_path=None):
        path_for_statistics = stats_path if stats_path else self.stats_path
        assert path_for_statistics, f"No path for statistics given. Either set it for specific class or provide it as param!"
        self.all_statistics.to_csv(str(path_for_statistics) + "/statistics.csv", index=False)
        print(f"Finished {inspect.currentframe().f_code.co_name}")

    def convert_all_to_polydata(self):
        assert self.has_loaded_data, f"No data was loaded. Run {self.load_files_in_memory.__name__} first!"
        self.full_data = [dict(**mesh_data, poly_data=pv.PolyData(mesh_data["data"]["vertices"], mesh_data["data"]["faces"])) for mesh_data in self.full_data]
        self.has_poly_data = True
        print(f"Finished {inspect.currentframe().f_code.co_name}")

    def detect_outliers(self):
        assert self.has_stats, f"No statistics were computed. Run {self.compute_shape_statistics.__name__} first"

        def add_to_stats(m_object):
            m_object["statistics"]["faces_outlier"] = m_object["faces_outlier"]
            m_object["statistics"]["vertices_outlier"] = m_object["vertices_outlier"]
            return m_object

        all_faces_counts, all_vertices_counts = self.all_statistics["faces"], self.all_statistics["vertices"]
        self.faces_mean, self.faces_std = all_faces_counts.mean(), all_faces_counts.std()
        self.vertices_mean, self.vertices_std = all_faces_counts.mean(), all_faces_counts.std()
        init_pipe = (mesh_object for mesh_object in self.full_data)
        faces_pipe = (dict(**mesh_object, faces_outlier=True if np.abs(mesh_object["statistics"]["faces"] - self.faces_mean) > 2 * self.faces_std else False)
                      for mesh_object in init_pipe)
        vertices_pipe = (dict(**mesh_object, vertices_outlier=True if np.abs(mesh_object["statistics"]["vertices"] - self.vertices_mean) > 2 * self.vertices_std else False)
                         for mesh_object in faces_pipe)
        add_to_stats_pipe = (add_to_stats(mesh_object) for mesh_object in vertices_pipe)

        self.full_data = list(add_to_stats_pipe)
        self.all_statistics = pd.DataFrame([mesh_object["statistics"] for mesh_object in self.full_data])
        print(f"Finished {inspect.currentframe().f_code.co_name}")

    def _compute_statistics(self, mesh):
        mesh_data = mesh["data"]
        poly_data_object = mesh["poly_data"]
        triangulized_poly_data_object = poly_data_object.triangulate()
        mesh["poly_data"] = triangulized_poly_data_object
        statistics = {}
        statistics["id"] = mesh["meta_data"]["name"]
        statistics["label"] = mesh["meta_data"]["label"]
        statistics["faces"] = poly_data_object.n_faces
        statistics["vertices"] = poly_data_object.n_points
        statistics.update(dict(zip(["bound_" + b for b in "xmin xmax ymin ymax zmin zmax".split()], poly_data_object.bounds)))
        cell_ids = self._get_cells(mesh["poly_data"])
        cell_point_counts = [len(cell) for cell in cell_ids]
        cell_counter = Counter(cell_point_counts)
        statistics.update({f"cell_type_{k}": v for k, v in cell_counter.items()})

        cell_areas = self._get_cell_areas(triangulized_poly_data_object.points, cell_ids, cell_point_counts)
        cell_centers = triangulized_poly_data_object.cell_centers().points
        mesh["bary_center"] = np.array(self._compute_center(cell_centers, cell_areas))
        statistics.update({f"center_{dim}": val for dim, val in zip("x y z".split(), mesh["bary_center"])})

        statistics["cell_area_mean"] = np.mean(cell_areas)
        statistics["cell_area_std"] = np.std(cell_areas)
        return statistics

    def _get_cell_areas(self, mesh_vertices, mesh_cells, mesh_cell_point_counts):
        cell_combinations = mesh_vertices[mesh_cells, :]
        # cell_areas = [np.abs(np.linalg.norm(np.cross((matrix[0] - matrix[1]), (matrix[0] - matrix[2])))) / 2 for matrix in cell_combinations]
        cell_areas_fast = np.abs(np.linalg.norm(np.cross((cell_combinations[:, 0] - cell_combinations[:, 1]),
                                                         (cell_combinations[:, 0] - cell_combinations[:, 2])), axis=1)) / 2  # https://math.stackexchange.com/a/128999

        return cell_areas_fast

    @staticmethod
    def _get_cells(mesh):
        """Returns a list of the cells from this mesh.
        This properly unpacks the VTK cells array.
        There are many ways to do this, but this is
        safe when dealing with mixed cell types."""

        return mesh.faces.reshape(-1, 4)[:, 1:4]
        # offset = 0
        # cells = []
        # for i in range(mesh.n_cells):
        #     loc = i + offset
        #     nc = mesh.faces[loc]
        #     offset += nc
        #     cell = mesh.faces[loc + 1:loc + nc + 1]
        #     cells.append(cell)
        # return cells

    def _load_ply(self, file):
        ply_data = PlyData.read(file)
        vertices = np.array(ply_data["vertex"].data.tolist())
        faces = ply_data["face"]["vertex_indices"].tolist()
        faces_with_number_of_faces = [np.hstack([face.shape[0], face]) for face in faces]
        flattened_faces = np.hstack(faces_with_number_of_faces).flatten()
        return {"vertices": vertices, "faces": flattened_faces}

    def _load_off(self, file):
        off_data = pv.read(file)
        faces = off_data.cells
        vertices = np.array(off_data.points)
        return {"vertices": vertices, "faces": faces}

    def _extract_descr(self, file_path):
        raise NotImplementedError

    def _compute_center(self, face_normals, face_areas):
        weighted_normals = face_areas.reshape(-1, 1) * face_normals
        bary_center = np.sum(weighted_normals, axis=0) / np.sum(face_areas)
        # return {f"center_{dim}" for dim, val in zip("x y z".split(), mesh.center)}
        # return {f"center_{dim}": val for dim, val in zip("x y z".split(), bary_center)}
        return bary_center

    def show_class_histogram(self):
        pd_data = pd.DataFrame(self.data_descriptors)
        counts_list = list(Counter(pd_data["label"].astype(str).values).items())
        counts = pd.DataFrame([{"label": label, "counts": int(count)} for label, count in counts_list])
        # counts = np.array()
        plt.bar(counts["label"], counts["counts"])
        plt.show()

    @staticmethod
    def show_barycenter(mesh_item):
        plotter = pv.Plotter()
        some_sphere = pv.Sphere(radius=0.001)
        some_sphere.translate(mesh_item["bary_center"])
        plotter.add_mesh(some_sphere)
        plotter.add_mesh(mesh_item["poly_data"])
        plotter.show()


class PSBDataset(DataSet):
    def __init__(self, search_path=None, stats_path=None, class_file_path=None):

        self.search_path = "data/psb" if not search_path else search_path
        # assert type(self.search_paths) == , f"Provide a list for the search paths not a {type(self.search_paths)}"
        self.search_paths = [Path(self.search_path) / scheme for scheme in self.schemes]  #if not self.search_paths else self.search_paths
        # assert self.search_paths, "No search paths given"
        self.stats_path = Path("data/psb") if not stats_path else Path(stats_path)
        self.class_file_path = class_file_path
        self.class_member_ships = self.load_classes()
        super().__init__(self.search_paths, self.stats_path)

    def load_classes(self):
        if not self.class_file_path:
            return {} 
        path_to_classes = Path(self.class_file_path)
        search_pattern = path_to_classes / "*.cla"
        class_files = list(glob.glob(str(search_pattern), recursive=True))
        class_file_handlers = [io.open(cfile, mode="r") for cfile in class_files]
        class_file_content = [cf.read().split("\n") for cf in class_file_handlers]
        curr_class = None
        class_memberships = {}
        for cfile in class_file_content:
            for line in cfile:
                line_content = line.split()
                line_nr_of_items = len(line_content)
                if line_nr_of_items == 2:
                    continue
                if line_nr_of_items == 3:
                    curr_class = line_content[0]
                if line_nr_of_items == 1:
                    class_memberships[f"m{line_content[0]}"] = curr_class
        return class_memberships

    def read(self):
        self.data_descriptors = [self._extract_descr(file_path) for file_path in self.data_file_paths]
        self.has_descriptors = True

    def _extract_descr(self, file_path):
        path = Path(file_path)
        file_name = path.stem
        file_type = path.suffix
        label = self.class_member_ships[file_name] if file_name in self.class_member_ships.keys() else "no_class"
        return {"label": label, "name": file_name, "type": file_type, "path": path.resolve()}


if __name__ == "__main__":
    # dataset = PSBDataset(search_path="D:\\Documents\\Programming\\Python\\project_multimedia_retrieval\\data", stats_path="stats")
    dataset = PSBDataset(DATA_PATH, class_file_path=CLASS_FILE)
    dataset.read()
    # dataset.show_class_histogram()
    dataset.load_files_in_memory()
    dataset.convert_all_to_polydata()
    dataset.compute_shape_statistics()
    dataset.detect_outliers()
    dataset.save_statistics()
    pprint(dataset.full_data[0])
