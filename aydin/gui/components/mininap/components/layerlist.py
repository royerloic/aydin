from ..layers import Layer

from ..util.naming import inc_name_count
from ..util.list import ListModel
from vispy.visuals.transforms import ChainTransform


def _add(event):
    """When a layer is added, set its name and order."""
    layers = event.source
    layer = event.item
    layer.name = layers._coerce_name(layer.name, layer)
    layer._order = -len(layers)
    layer.events.name.connect(lambda e: layers._update_name(e))
    layers.unselect_all(ignore=layer)


def _remove(event):
    """When a layer is removed, remove its viewer."""
    layer = event.item
    layer._order = 0
    layer._node.transforms = ChainTransform()
    layer._node.parent = None


def _reorder(event):
    """When the list is reordered, propagate those changes to draw order."""
    layers = event.source
    for i in range(len(layers)):
        layers[i]._order = -i


class LayerList(ListModel):
    """List-like layer collection with built-in reordering and callback hooks.

    Attributes
    ----------
    events : vispy.util.event.EmitterGroup
        Event hooks:
            * added(item, index): whenever an item is added
            * removed(item): whenever an item is removed
            * reordered(): whenever the list is reordered
    """

    def __init__(self):
        super().__init__(basetype=Layer, lookup={str: lambda q, e: q == e.name})

        self.events.added.connect(_add)
        self.events.removed.connect(_remove)
        self.events.reordered.connect(_reorder)

    def __newlike__(self, iterable):
        return ListModel(self._basetype, iterable, self._lookup)

    def _coerce_name(self, name, layer=None):
        """Coerce a name into a unique equivalent.

        Parameters
        ----------
        name : str
            Original name.
        layer : Layer, optional
            Layer for which name is generated.

        Returns
        -------
        new_name : str
            Coerced, unique name.
        """
        for l in self:
            if l is layer:
                continue
            if l.name == name:
                name = inc_name_count(name)

        return name

    def _update_name(self, event):
        """Coerce name of the layer in `event.layer`."""
        layer = event.source
        layer.name = self._coerce_name(layer.name, layer)

    def move_selected(self, index, insert):
        """Reorder list by moving the item at index and inserting it
        at the insert index. If additional items are selected these will
        get inserted at the insert index too. This allows for rearranging
        the list based on dragging and dropping a selection of items, where
        index is the index of the primary item being dragged, and insert is
        the index of the drop location, and the selection indicates if
        multiple items are being dragged. If the moved layer is not selected
        select it.

        Parameters
        ----------
        index : int
            Index of primary item to be moved
        insert : int
            Index that item(s) will be inserted at
        """
        total = len(self)
        indices = list(range(total))
        if not self[index].selected:
            self.unselect_all()
            self[index].selected = True
        selected = [i for i in range(total) if self[i].selected]

        # remove all indices to be moved
        for i in selected:
            indices.remove(i)
        # adjust offset based on selected indices to move
        offset = sum([i < insert and i != index for i in selected])
        # insert indices to be moved at correct start
        for insert_idx, elem_idx in enumerate(selected, start=insert - offset):
            indices.insert(insert_idx, elem_idx)
        # reorder list
        self[:] = self[tuple(indices)]

    def unselect_all(self, ignore=None):
        """Unselects all layers expect any specified in ignore.

        Parameters
        ----------
        ignore : Layer | None
            Layer that should not be unselected if specified.
        """
        for layer in self:
            if layer.selected and layer != ignore:
                layer.selected = False

    def remove_selected(self):
        """Removes selected items from list.
        """
        to_delete = []
        for i in range(len(self)):
            if self[i].selected:
                to_delete.append(i)
        to_delete.reverse()
        for i in to_delete:
            self.pop(i)
