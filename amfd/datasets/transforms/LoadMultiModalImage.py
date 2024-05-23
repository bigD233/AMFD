from mmdet.registry import TRANSFORMS
import warnings
from typing import Optional
from mmengine.registry import TRANSFORMS as MMCV_TRANSFORMS
import mmengine.fileio as fileio
import numpy as np
from mmcv.transforms import BaseTransform
import mmcv
from mmdet.datasets import CocoDataset
from mmdet.structures.bbox import get_box_type
from mmcv.transforms import LoadAnnotations as MMCV_LoadAnnotations
import cv2, os
import torch
from typing import List, Optional, Sequence, Tuple, Union
from numbers import Number
from mmengine.utils import is_seq_of
import math
from mmdet.registry import MODELS
import torch.nn as nn
import torch.nn.functional as F


@TRANSFORMS.register_module()
class LoadBGR3TFromKAIST(BaseTransform):
    """Load an image from file.

    Required Keys:

    - img_path

    Modified Keys:

    - img
    - img_shape
    - ori_shape

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        imdecode_backend (str): The image decoding backend type. The backend
            argument for :func:`mmcv.imfrombytes`.
            See :func:`mmcv.imfrombytes` for details.
            Defaults to 'cv2'.
        file_client_args (dict, optional): Arguments to instantiate a
            FileClient. See :class:`mmengine.fileio.FileClient` for details.
            Defaults to None. It will be deprecated in future. Please use
            ``backend_args`` instead.
            Deprecated in version 2.0.0rc4.
        ignore_empty (bool): Whether to allow loading empty image or file path
            not existent. Defaults to False.
        backend_args (dict, optional): Instantiates the corresponding file
            backend. It may contain `backend` key to specify the file
            backend. If it contains, the file backend corresponding to this
            value will be used and initialized with the remaining values,
            otherwise the corresponding file backend will be selected
            based on the prefix of the file path. Defaults to None.
            New in version 2.0.0rc4.
    """

    def __init__(self,
                 to_float32: bool = False,
                 color_type: str = 'color',
                 imdecode_backend: str = 'cv2',
                 file_client_args: Optional[dict] = None,
                 ignore_empty: bool = False,
                 *,
                 backend_args: Optional[dict] = None) -> None:
        self.ignore_empty = ignore_empty
        self.to_float32 = to_float32
        self.color_type = color_type
        self.imdecode_backend = imdecode_backend

        self.file_client_args: Optional[dict] = None
        self.backend_args: Optional[dict] = None
        if file_client_args is not None:
            warnings.warn(
                '"file_client_args" will be deprecated in future. '
                'Please use "backend_args" instead', DeprecationWarning)
            if backend_args is not None:
                raise ValueError(
                    '"file_client_args" and "backend_args" cannot be set '
                    'at the same time.')

            self.file_client_args = file_client_args.copy()
        if backend_args is not None:
            self.backend_args = backend_args.copy()

    def transform(self, results: dict) -> Optional[dict]:
        """Functions to load image.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        if 'visible' in results['img_path']:
            filename_rgb = results['img_path']
            if 'train' in filename_rgb:
                path_prefix=filename_rgb.split('visible')[0]
                image_name = filename_rgb.split('visible')[1]
                filename_ir=path_prefix+'lwir'+image_name
            else: 
                filename_rgb = filename_rgb + '_visible.png'
                # filename_ir = filename_rgb.replace('visible', 'lwir')
                filename_rgb = os.path.dirname(filename_ir).replace('lwir', 'visible') + '_complex_light_new/' + \
                    os.path.basename(filename_ir).replace('lwir', 'visible')
        else:
            filename_ir = results['img_path']
            if 'train' in filename_ir:
                path_prefix=filename_ir.split('lwir')[0]
                image_name = filename_ir.split('lwir')[1]
                filename_rgb=path_prefix+'visible'+image_name
            else: 
                if 'complex' not in filename_ir:
                    filename_ir = filename_ir + '_lwir.png'
                    filename_rgb = filename_ir.replace('lwir', 'visible')
                else:
                    filename_ir = filename_ir.replace("complex_light_new_", "") + '_lwir.png'
                    filename_rgb = os.path.dirname(filename_ir).replace('lwir', 'visible') + '_complex_light_new/' + \
                        os.path.basename(filename_ir).replace('lwir', 'visible')
        try:
            if self.file_client_args is not None:
                file_client_ir = fileio.FileClient.infer_client(
                    self.file_client_args, filename_ir)
                img_bytes_ir = file_client_ir.get(filename_ir)
                file_client_rgb = fileio.FileClient.infer_client(
                    self.file_client_args, filename_rgb)
                img_bytes_rgb = file_client_rgb.get(filename_rgb)
            else:
                img_bytes_ir = fileio.get(
                    filename_ir, backend_args=self.backend_args)
                img_bytes_rgb = fileio.get(
                    filename_rgb, backend_args=self.backend_args)
            img_ir = mmcv.imfrombytes(
                img_bytes_ir, flag=self.color_type, backend=self.imdecode_backend)
            
            # try_img=cv2.imread(filename_rgb)
            img_rgb = mmcv.imfrombytes(
                img_bytes_rgb, flag=self.color_type, backend=self.imdecode_backend)
            
            height, width, _ = img_rgb.shape # 512, 640
            img = np.zeros((height, width, 6), dtype=np.uint8)
            # print(try_img==img_rgb) 
            img[:, :, :3] = img_rgb
            img[:, :, 3:] = img_ir
            # print(img.shape)

        except Exception as e:
            if self.ignore_empty:
                return None
            else:
                raise e
        # in some cases, images are not read successfully, the img would be
        # `None`, refer to https://github.com/open-mmlab/mmpretrain/issues/1427
        assert img is not None, f'failed to load image: {filename_ir}'
        if self.to_float32:
            img = img.astype(np.float32)
       
        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'ignore_empty={self.ignore_empty}, '
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f"imdecode_backend='{self.imdecode_backend}', ")

        if self.file_client_args is not None:
            repr_str += f'file_client_args={self.file_client_args})'
        else:
            repr_str += f'backend_args={self.backend_args})'

        return repr_str


@TRANSFORMS.register_module()
class LoadBGR3TFromFLIR(BaseTransform):
    """Load an image from file.

    Required Keys:

    - img_path

    Modified Keys:

    - img
    - img_shape
    - ori_shape

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        imdecode_backend (str): The image decoding backend type. The backend
            argument for :func:`mmcv.imfrombytes`.
            See :func:`mmcv.imfrombytes` for details.
            Defaults to 'cv2'.
        file_client_args (dict, optional): Arguments to instantiate a
            FileClient. See :class:`mmengine.fileio.FileClient` for details.
            Defaults to None. It will be deprecated in future. Please use
            ``backend_args`` instead.
            Deprecated in version 2.0.0rc4.
        ignore_empty (bool): Whether to allow loading empty image or file path
            not existent. Defaults to False.
        backend_args (dict, optional): Instantiates the corresponding file
            backend. It may contain `backend` key to specify the file
            backend. If it contains, the file backend corresponding to this
            value will be used and initialized with the remaining values,
            otherwise the corresponding file backend will be selected
            based on the prefix of the file path. Defaults to None.
            New in version 2.0.0rc4.
    """

    def __init__(self,
                 to_float32: bool = False,
                 color_type: str = 'color',
                 imdecode_backend: str = 'cv2',
                 file_client_args: Optional[dict] = None,
                 ignore_empty: bool = False,
                 *,
                 backend_args: Optional[dict] = None) -> None:
        self.ignore_empty = ignore_empty
        self.to_float32 = to_float32
        self.color_type = color_type
        self.imdecode_backend = imdecode_backend

        self.file_client_args: Optional[dict] = None
        self.backend_args: Optional[dict] = None
        if file_client_args is not None:
            warnings.warn(
                '"file_client_args" will be deprecated in future. '
                'Please use "backend_args" instead', DeprecationWarning)
            if backend_args is not None:
                raise ValueError(
                    '"file_client_args" and "backend_args" cannot be set '
                    'at the same time.')

            self.file_client_args = file_client_args.copy()
        if backend_args is not None:
            self.backend_args = backend_args.copy()

    def transform(self, results: dict) -> Optional[dict]:
        """Functions to load image.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        filename_ir = results['img_path']
        path_prefix=filename_ir.split('thermal_8_bit')[0]
        image_name = filename_ir.split('thermal_8_bit')[1][:-4]
        filename_rgb=path_prefix+'RGB'+image_name+'jpg'

        try:
            if self.file_client_args is not None:
                file_client_ir = fileio.FileClient.infer_client(
                    self.file_client_args, filename_ir)
                img_bytes_ir = file_client_ir.get(filename_ir)
                file_client_rgb = fileio.FileClient.infer_client(
                    self.file_client_args, filename_rgb)
                img_bytes_rgb = file_client_rgb.get(filename_rgb)
            else:
                img_bytes_ir = fileio.get(
                    filename_ir, backend_args=self.backend_args)
                img_bytes_rgb = fileio.get(
                    filename_rgb, backend_args=self.backend_args)
            img_ir = mmcv.imfrombytes(
                img_bytes_ir, flag=self.color_type, backend=self.imdecode_backend)
            
            # try_img=cv2.imread(filename_rgb)
            img_rgb = mmcv.imfrombytes(
                img_bytes_rgb, flag=self.color_type, backend=self.imdecode_backend)
            
            height, width, _ = img_ir.shape
            img = np.zeros((height, width, 6), dtype=np.uint8)
            img_rgb = mmcv.imresize(img_rgb, (img_ir.shape[1], img_ir.shape[0]))

            img[:, :, :3] = img_rgb
            img[:, :, 3:] = img_ir


        except Exception as e:
            if self.ignore_empty:
                return None
            else:
                raise e
        # in some cases, images are not read successfully, the img would be
        # `None`, refer to https://github.com/open-mmlab/mmpretrain/issues/1427
        assert img is not None, f'failed to load image: {filename_ir}'
        if self.to_float32:
            img = img.astype(np.float32)
       
        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'ignore_empty={self.ignore_empty}, '
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f"imdecode_backend='{self.imdecode_backend}', ")

        if self.file_client_args is not None:
            repr_str += f'file_client_args={self.file_client_args})'
        else:
            repr_str += f'backend_args={self.backend_args})'

        return repr_str

@TRANSFORMS.register_module()
class LoadBGR3TFromLLVIP(BaseTransform):
    """Load an image from file.

    Required Keys:

    - img_path

    Modified Keys:

    - img
    - img_shape
    - ori_shape

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        imdecode_backend (str): The image decoding backend type. The backend
            argument for :func:`mmcv.imfrombytes`.
            See :func:`mmcv.imfrombytes` for details.
            Defaults to 'cv2'.
        file_client_args (dict, optional): Arguments to instantiate a
            FileClient. See :class:`mmengine.fileio.FileClient` for details.
            Defaults to None. It will be deprecated in future. Please use
            ``backend_args`` instead.
            Deprecated in version 2.0.0rc4.
        ignore_empty (bool): Whether to allow loading empty image or file path
            not existent. Defaults to False.
        backend_args (dict, optional): Instantiates the corresponding file
            backend. It may contain `backend` key to specify the file
            backend. If it contains, the file backend corresponding to this
            value will be used and initialized with the remaining values,
            otherwise the corresponding file backend will be selected
            based on the prefix of the file path. Defaults to None.
            New in version 2.0.0rc4.
    """

    def __init__(self,
                 to_float32: bool = False,
                 color_type: str = 'color',
                 imdecode_backend: str = 'cv2',
                 file_client_args: Optional[dict] = None,
                 ignore_empty: bool = False,
                 *,
                 backend_args: Optional[dict] = None) -> None:
        self.ignore_empty = ignore_empty
        self.to_float32 = to_float32
        self.color_type = color_type
        self.imdecode_backend = imdecode_backend

        self.file_client_args: Optional[dict] = None
        self.backend_args: Optional[dict] = None
        if file_client_args is not None:
            warnings.warn(
                '"file_client_args" will be deprecated in future. '
                'Please use "backend_args" instead', DeprecationWarning)
            if backend_args is not None:
                raise ValueError(
                    '"file_client_args" and "backend_args" cannot be set '
                    'at the same time.')

            self.file_client_args = file_client_args.copy()
        if backend_args is not None:
            self.backend_args = backend_args.copy()

    def transform(self, results: dict) -> Optional[dict]:
        """Functions to load image.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        filename_ir = results['img_path']
        filename_rgb = filename_ir.replace('infrared','visible')

        try:
            if self.file_client_args is not None:
                file_client_ir = fileio.FileClient.infer_client(
                    self.file_client_args, filename_ir)
                img_bytes_ir = file_client_ir.get(filename_ir)
                file_client_rgb = fileio.FileClient.infer_client(
                    self.file_client_args, filename_rgb)
                img_bytes_rgb = file_client_rgb.get(filename_rgb)
            else:
                img_bytes_ir = fileio.get(
                    filename_ir, backend_args=self.backend_args)
                img_bytes_rgb = fileio.get(
                    filename_rgb, backend_args=self.backend_args)
            img_ir = mmcv.imfrombytes(
                img_bytes_ir, flag=self.color_type, backend=self.imdecode_backend)
            
            # try_img=cv2.imread(filename_rgb)
            img_rgb = mmcv.imfrombytes(
                img_bytes_rgb, flag=self.color_type, backend=self.imdecode_backend)
            
            height, width, _ = img_rgb.shape
            img = np.zeros((height, width, 6), dtype=np.uint8)
            # print(try_img==img_rgb) 
            img[:, :, :3] = img_rgb
            img[:, :, 3:] = img_ir
            # print(img.shape)

        except Exception as e:
            if self.ignore_empty:
                return None
            else:
                raise e
        # in some cases, images are not read successfully, the img would be
        # `None`, refer to https://github.com/open-mmlab/mmpretrain/issues/1427
        assert img is not None, f'failed to load image: {filename_ir}'
        if self.to_float32:
            img = img.astype(np.float32)
       
        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'ignore_empty={self.ignore_empty}, '
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f"imdecode_backend='{self.imdecode_backend}', ")

        if self.file_client_args is not None:
            repr_str += f'file_client_args={self.file_client_args})'
        else:
            repr_str += f'backend_args={self.backend_args})'

        return repr_str

@TRANSFORMS.register_module()
class LoadBGR3TFromM3FD(BaseTransform):
    """Load an image from file.

    Required Keys:

    - img_path

    Modified Keys:

    - img
    - img_shape
    - ori_shape

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        imdecode_backend (str): The image decoding backend type. The backend
            argument for :func:`mmcv.imfrombytes`.
            See :func:`mmcv.imfrombytes` for details.
            Defaults to 'cv2'.
        file_client_args (dict, optional): Arguments to instantiate a
            FileClient. See :class:`mmengine.fileio.FileClient` for details.
            Defaults to None. It will be deprecated in future. Please use
            ``backend_args`` instead.
            Deprecated in version 2.0.0rc4.
        ignore_empty (bool): Whether to allow loading empty image or file path
            not existent. Defaults to False.
        backend_args (dict, optional): Instantiates the corresponding file
            backend. It may contain `backend` key to specify the file
            backend. If it contains, the file backend corresponding to this
            value will be used and initialized with the remaining values,
            otherwise the corresponding file backend will be selected
            based on the prefix of the file path. Defaults to None.
            New in version 2.0.0rc4.
    """

    def __init__(self,
                 to_float32: bool = False,
                 color_type: str = 'color',
                 imdecode_backend: str = 'cv2',
                 file_client_args: Optional[dict] = None,
                 ignore_empty: bool = False,
                 *,
                 backend_args: Optional[dict] = None) -> None:
        self.ignore_empty = ignore_empty
        self.to_float32 = to_float32
        self.color_type = color_type
        self.imdecode_backend = imdecode_backend

        self.file_client_args: Optional[dict] = None
        self.backend_args: Optional[dict] = None
        if file_client_args is not None:
            warnings.warn(
                '"file_client_args" will be deprecated in future. '
                'Please use "backend_args" instead', DeprecationWarning)
            if backend_args is not None:
                raise ValueError(
                    '"file_client_args" and "backend_args" cannot be set '
                    'at the same time.')

            self.file_client_args = file_client_args.copy()
        if backend_args is not None:
            self.backend_args = backend_args.copy()

    def transform(self, results: dict) -> Optional[dict]:
        """Functions to load image.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        if 'Ir' in results['img_path']:
            filename_ir = results['img_path']
            filename_rgb = filename_ir.replace('Ir', 'Vis')
        
        try:
            if self.file_client_args is not None:
                file_client_ir = fileio.FileClient.infer_client(
                    self.file_client_args, filename_ir)
                img_bytes_ir = file_client_ir.get(filename_ir)
                file_client_rgb = fileio.FileClient.infer_client(
                    self.file_client_args, filename_rgb)
                img_bytes_rgb = file_client_rgb.get(filename_rgb)
            else:
                img_bytes_ir = fileio.get(
                    filename_ir, backend_args=self.backend_args)
                img_bytes_rgb = fileio.get(
                    filename_rgb, backend_args=self.backend_args)
            img_ir = mmcv.imfrombytes(
                img_bytes_ir, flag=self.color_type, backend=self.imdecode_backend)
            
            # try_img=cv2.imread(filename_rgb)
            img_rgb = mmcv.imfrombytes(
                img_bytes_rgb, flag=self.color_type, backend=self.imdecode_backend)
            
            height, width, _ = img_rgb.shape # 512, 640
            img = np.zeros((height, width, 6), dtype=np.uint8)
            # print(try_img==img_rgb) 
            img[:, :, :3] = img_rgb
            img[:, :, 3:] = img_ir
            # print(img.shape)

        except Exception as e:
            if self.ignore_empty:
                return None
            else:
                raise e
        # in some cases, images are not read successfully, the img would be
        # `None`, refer to https://github.com/open-mmlab/mmpretrain/issues/1427
        assert img is not None, f'failed to load image: {filename_ir}'
        if self.to_float32:
            img = img.astype(np.float32)
       
        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'ignore_empty={self.ignore_empty}, '
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f"imdecode_backend='{self.imdecode_backend}', ")

        if self.file_client_args is not None:
            repr_str += f'file_client_args={self.file_client_args})'
        else:
            repr_str += f'backend_args={self.backend_args})'

        return repr_str



@TRANSFORMS.register_module()
@MMCV_TRANSFORMS.register_module()
class Normalize_Pad(BaseTransform):
    def __init__(self,
                 mean: Sequence[Number] = None,
                 std: Sequence[Number] = None,
                 pad_size_divisor: int = 1,
                 pad_value: Union[float, int] = 0,
                 pad_mask: bool = False,
                 mask_pad_value: int = 0,
                 pad_seg: bool = False,
                 seg_pad_value: int = 255,
                 bgr_to_rgb: bool = False,
                 rgb_to_bgr: bool = False,
                 boxtype2tensor: bool = True,
                 non_blocking: Optional[bool] = False,
                 batch_augments: Optional[List[dict]] = None):
        # super().__init__(non_blocking)
        
        self.pad_size_divisor=pad_size_divisor
        self.pad_value=pad_value
        self.bgr_to_rgb=bgr_to_rgb
        self.rgb_to_bgr=rgb_to_bgr
        self.non_blocking=non_blocking
        if batch_augments is not None:
            self.batch_augments = nn.ModuleList(
                [MODELS.build(aug) for aug in batch_augments])
        else:
            self.batch_augments = None
        self.pad_mask = pad_mask
        self.mask_pad_value = mask_pad_value
        self.pad_seg = pad_seg
        self.seg_pad_value = seg_pad_value
        self.boxtype2tensor = boxtype2tensor

        if len(mean)==6 and len(std)==6:
            self.register_buffer('mean',
                                 torch.tensor(mean).view(-1, 1, 1), False)
            self.register_buffer('std',
                                 torch.tensor(std).view(-1, 1, 1), False)
    
    def transform(self, data: dict) -> dict:

        # transform bgrttt image according to mean and std (shape=6)
        data = self.cast_data(data)  # type: ignore
        _batch_inputs = data['img']
        # Process data with `pseudo_collate`.
        if is_seq_of(_batch_inputs, torch.Tensor):
            batch_inputs = []
            for _batch_input in _batch_inputs:
                
                _batch_input = _batch_input.float()
                # Normalization.
                # print(_batch_input,self.mean)
                _batch_input = (_batch_input - self.mean) / self.std
                
                batch_inputs.append(_batch_input)
            # Pad and stack Tensor.
            batch_inputs = self.stack_batch(batch_inputs, self.pad_size_divisor,
                                       self.pad_value)
        # Process data with `default_collate`.
        elif isinstance(_batch_inputs, torch.Tensor):
            assert _batch_inputs.dim() == 4, (
                'The input of `ImgDataPreprocessor` should be a NCHW tensor '
                'or a list of tensor, but got a tensor with shape: '
                f'{_batch_inputs.shape}')
            _batch_inputs = _batch_inputs.float()
            _batch_inputs = (_batch_inputs - self.mean) / self.std
            h, w = _batch_inputs.shape[2:]
            target_h = math.ceil(
                h / self.pad_size_divisor) * self.pad_size_divisor
            target_w = math.ceil(
                w / self.pad_size_divisor) * self.pad_size_divisor
            pad_h = target_h - h
            pad_w = target_w - w
            batch_inputs = F.pad(_batch_inputs, (0, pad_w, 0, pad_h),
                                 'constant', self.pad_value)
        else:
            raise TypeError('Output of `cast_data` should be a dict of '
                            'list/tuple with inputs and data_samples, '
                            f'but got {type(data)}: {data}')
        data['img'] = batch_inputs
        # data.setdefault('data_samples', None)
        return data


@TRANSFORMS.register_module()
class LoadBGR3TFromSMCT(BaseTransform):
    """Load an image from file.

    Required Keys:

    - img_path

    Modified Keys:

    - img
    - img_shape
    - ori_shape

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        imdecode_backend (str): The image decoding backend type. The backend
            argument for :func:`mmcv.imfrombytes`.
            See :func:`mmcv.imfrombytes` for details.
            Defaults to 'cv2'.
        file_client_args (dict, optional): Arguments to instantiate a
            FileClient. See :class:`mmengine.fileio.FileClient` for details.
            Defaults to None. It will be deprecated in future. Please use
            ``backend_args`` instead.
            Deprecated in version 2.0.0rc4.
        ignore_empty (bool): Whether to allow loading empty image or file path
            not existent. Defaults to False.
        backend_args (dict, optional): Instantiates the corresponding file
            backend. It may contain `backend` key to specify the file
            backend. If it contains, the file backend corresponding to this
            value will be used and initialized with the remaining values,
            otherwise the corresponding file backend will be selected
            based on the prefix of the file path. Defaults to None.
            New in version 2.0.0rc4.
    """

    def __init__(self,
                 to_float32: bool = False,
                 color_type: str = 'color',
                 imdecode_backend: str = 'cv2',
                 file_client_args: Optional[dict] = None,
                 ignore_empty: bool = False,
                 *,
                 backend_args: Optional[dict] = None) -> None:
        self.ignore_empty = ignore_empty
        self.to_float32 = to_float32
        self.color_type = color_type
        self.imdecode_backend = imdecode_backend

        self.file_client_args: Optional[dict] = None
        self.backend_args: Optional[dict] = None
        if file_client_args is not None:
            warnings.warn(
                '"file_client_args" will be deprecated in future. '
                'Please use "backend_args" instead', DeprecationWarning)
            if backend_args is not None:
                raise ValueError(
                    '"file_client_args" and "backend_args" cannot be set '
                    'at the same time.')

            self.file_client_args = file_client_args.copy()
        if backend_args is not None:
            self.backend_args = backend_args.copy()

    def transform(self, results: dict) -> Optional[dict]:
        """Functions to load image.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        filename_prefix,file_name_ori  = results['img_path'].split("/0")[0], results['img_path'].split("/0")[1]
        filename_id = file_name_ori.split("_rgb")[0]
        filename_ir = filename_prefix + "/TIR" +"/0"+ filename_id + ".jpg"
        filename_rgb = filename_prefix + "/RGB" + "/0"+ filename_id + ".jpg"
            
        try:
            if self.file_client_args is not None:
                file_client_ir = fileio.FileClient.infer_client(
                    self.file_client_args, filename_ir)
                img_bytes_ir = file_client_ir.get(filename_ir)
                file_client_rgb = fileio.FileClient.infer_client(
                    self.file_client_args, filename_rgb)
                img_bytes_rgb = file_client_rgb.get(filename_rgb)
            else:
                img_bytes_ir = fileio.get(
                    filename_ir, backend_args=self.backend_args)
                img_bytes_rgb = fileio.get(
                    filename_rgb, backend_args=self.backend_args)
            img_ir = mmcv.imfrombytes(
                img_bytes_ir, flag=self.color_type, backend=self.imdecode_backend)
            
            # try_img=cv2.imread(filename_rgb)
            img_rgb = mmcv.imfrombytes(
                img_bytes_rgb, flag=self.color_type, backend=self.imdecode_backend)
            
            height, width, _ = img_rgb.shape # 512, 640
            img = np.zeros((height, width, 6), dtype=np.uint8)
            # print(try_img==img_rgb) 
            img[:, :, :3] = img_rgb
            img[:, :, 3:] = img_ir
            # print(img.shape)

        except Exception as e:
            if self.ignore_empty:
                return None
            else:
                raise e
        # in some cases, images are not read successfully, the img would be
        # `None`, refer to https://github.com/open-mmlab/mmpretrain/issues/1427
        assert img is not None, f'failed to load image: {filename_ir}'
        if self.to_float32:
            img = img.astype(np.float32)
       
        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'ignore_empty={self.ignore_empty}, '
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f"imdecode_backend='{self.imdecode_backend}', ")

        if self.file_client_args is not None:
            repr_str += f'file_client_args={self.file_client_args})'
        else:
            repr_str += f'backend_args={self.backend_args})'

        return repr_str




