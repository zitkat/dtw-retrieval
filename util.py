#!python
# -*- coding: utf-8 -*-
"""General utils, some of them unused in this project."""

__author__ = "Tomas Zitka"
__email__ = "zitkat@kky.zcu.cz"

import re
import time
from pathlib import Path
from typing import Dict, Any

import unicodedata
from PIL.Image import Image as PILImage

import click
import ast
from functools import singledispatch, update_wrapper, lru_cache

from PIL import Image


def make_path(*pathargs, isdir=False, **pathkwargs):
    new_path = Path(*pathargs, **pathkwargs)
    return ensured_path(new_path, isdir=isdir)


def ensured_path(path: Path, isdir=False):
    if isdir:
        path.mkdir(parents=True, exist_ok=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


def now():
    """
    :return: date and time as YYYY-mm-dd-hh-MM
    """
    return time.strftime("%Y-%m-%d-%H-%M")


class PythonLiteralOption(click.Option):

    def type_cast_value(self, ctx, value):
        try:
            return ast.literal_eval(value)
        except:
            raise click.BadParameter(value)


def methdispatch(func):
    dispatcher = singledispatch(func)

    def wrapper(*args, **kw):
        return dispatcher.dispatch(args[1].__class__)(*args, **kw)

    wrapper.register = dispatcher.register
    update_wrapper(wrapper, func)
    return wrapper


def load_img_withrotation(file_path: Path) -> PILImage:
    in_img = Image.open(str(file_path))
    exif = in_img._getexif()
    if exif is not None:
        orientation_tag = 274
        # from PIL.ExifTags.TAGS
        # [(k, v) for k, v in  PIL.ExifTags.TAGS.items() if v in ["Orientation"]
        if exif.get(orientation_tag, None) == 3:
            in_img = in_img.rotate(180, expand=True)
        elif exif.get(orientation_tag, None) == 6:
            in_img = in_img.rotate(270, expand=True)
        elif exif.get(orientation_tag, None) == 8:
            in_img = in_img.rotate(90, expand=True)


    return in_img


def slugify(value, allow_unicode=False):
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value)
    return re.sub(r'[-\s]+', '-', value).strip('-_')


def report(*args, end="\n"):
    print(*args, end=end)
    return args[0] if len(args) == 1 else args


def tupleasdict(tup, keys=None) -> Dict[Any, Any]:
    if keys is None:
        keys = range(len(tup))
    if len(keys) != len(tup):
        raise ValueError("Number of keys must be the same as tuple length")
    return {k: t for k, t in zip(keys, tup)}


def f_and(*fargs):
    return lambda *args, **kwargs: all(f(*args, **kwargs) for f in fargs)


def f_or(*fargs):
    return lambda *args, **kwargs: any(f(*args, **kwargs) for f in fargs)


def partial_keywords(f, **fkwargs):
    return lambda *args, **kwargs: f(*args, **{**fkwargs, **kwargs})
