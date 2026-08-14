"""Visual Flowgraph Editor for DSP Blocks using PyQt6."""

import sys
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
import logging

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGraphicsScene, QGraphicsView, QGraphicsItem, QGraphicsRectItem,
    QGraphicsEllipseItem, QGraphicsLineItem, QGraphicsTextItem,
    QMenu, QMenuBar, QDockWidget, QListWidget, QListWidgetItem,
    QDialog, QLabel, QLineEdit, QSpinBox, QDoubleSpinBox, QPushButton,
    QFormLayout, QComboBox, QStatusBar, QFileDialog, QMessageBox
)
from PyQt6.QtCore import (
    Qt, QPointF, QRectF, QSize, pyqtSignal, QObject, QTimer
)
from PyQt6.QtGui import (
    QColor, QPen, QBrush, QFont, QPixmap, QIcon, QPainter, QPolygon
)

logger = logging.getLogger(__name__)


class BlockType(Enum):
    """Types of DSP blocks."""
    SOURCE = "source"
    FILTER = "filter"
    MODULATOR = "modulator"
    DEMODULATOR = "demodulator"
    DETECTOR = "detector"
    SINK = "sink"


class PortType(Enum):
    """Port types for block connections."""
    INPUT = "in"
    OUTPUT = "out"


@dataclass
class PortConfig:
    """Configuration for a port."""
    name: str
    type: PortType
    data_type: str = "complex"  # 'complex', 'float', 'int'


@dataclass
class BlockConfig:
    """Configuration for a DSP block."""
    id: str
    name: str
    block_type: BlockType
    x: float = 0.0
    y: float = 0.0
    params: Dict = field(default_factory=dict)
    ports: List[PortConfig] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        d = asdict(self)
        d['block_type'] = self.block_type.value
        d['ports'] = [
            {
                'name': p.name,
                'type': p.type.value,
                'data_type': p.data_type
            }
            for p in d['ports']
        ]
        return d


@dataclass
class ConnectionConfig:
    """Configuration for a connection between blocks."""
    source_block: str
    source_port: str
    target_block: str
    target_port: str


class Port(QGraphicsEllipseItem):
    """Represents a port on a DSP block."""
    
    RADIUS = 5
    
    def __init__(self, block: 'DSPBlock', config: PortConfig, index: int):
        """Initialize port.
        
        Args:
            block: Parent DSP block
            config: Port configuration
            index: Port index for positioning
        """
        super().__init__()
        self.block = block
        self.config = config
        self.index = index
        self.connections: List['Connection'] = []
        
        # Setup appearance
        self.setRect(-self.RADIUS, -self.RADIUS, 2*self.RADIUS, 2*self.RADIUS)
        color = QColor(0, 200, 0) if config.type == PortType.INPUT else QColor(200, 0, 0)
        self.setBrush(QBrush(color))
        self.setPen(QPen(QColor(0, 0, 0)))
        
        # Position relative to block
        if config.type == PortType.INPUT:
            x = -block.width / 2
        else:
            x = block.width / 2
        
        y = -block.height / 2 + 15 + index * 20
        self.setPos(x, y)
    
    def get_center(self) -> QPointF:
        """Get absolute center position of port."""
        return self.scenePos() + QPointF(self.RADIUS, self.RADIUS)


class DSPBlock(QGraphicsRectItem):
    """Represents a DSP processing block."""
    
    width = 100
    height = 80
    
    def __init__(self, config: BlockConfig):
        """Initialize DSP block.
        
        Args:
            config: Block configuration
        """
        super().__init__()
        self.config = config
        self.ports: Dict[str, Port] = {}
        self.selected = False
        
        # Setup appearance
        self.setRect(-self.width/2, -self.height/2, self.width, self.height)
        self.set_color()
        self.setPos(config.x, config.y)
        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        
        # Add text label
        self.label = QGraphicsTextItem(self)
        self.label.setPlainText(config.name)
        font = QFont("Arial", 10, QFont.Weight.Bold)
        self.label.setFont(font)
        self.label.setPos(-self.width/2 + 5, -self.height/2 + 5)
        
        # Create ports
        self.create_ports()
    
    def set_color(self):
        """Set block color based on type."""
        colors = {
            BlockType.SOURCE: QColor(100, 200, 100),
            BlockType.FILTER: QColor(100, 100, 200),
            BlockType.MODULATOR: QColor(200, 100, 100),
            BlockType.DEMODULATOR: QColor(200, 200, 100),
            BlockType.DETECTOR: QColor(100, 200, 200),
            BlockType.SINK: QColor(200, 100, 200)
        }
        color = colors.get(self.config.block_type, QColor(150, 150, 150))
        self.setBrush(QBrush(color))
        self.setPen(QPen(QColor(0, 0, 0), 2))
    
    def create_ports(self):
        """Create input and output ports."""
        for i, port_config in enumerate(self.config.ports):
            port = Port(self, port_config, i)
            port.setParentItem(self)
            self.ports[port_config.name] = port
    
    def mouseMoveEvent(self, event):
        """Handle mouse move for dragging."""
        super().mouseMoveEvent(event)
        # Update connected lines
        for port in self.ports.values():
            for connection in port.connections:
                connection.update_path()
    
    def itemChange(self, change, value):
        """Handle item changes."""
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            # Update block config position
            self.config.x = self.pos().x()
            self.config.y = self.pos().y()
            # Update connected lines
            for port in self.ports.values():
                for connection in port.connections:
                    connection.update_path()
        
        return super().itemChange(change, value)


class Connection(QGraphicsLineItem):
    """Represents a connection between two ports."""
    
    def __init__(self, source_port: Port, target_port: Port):
        """Initialize connection.
        
        Args:
            source_port: Source port
            target_port: Target port
        """
        super().__init__()
        self.source_port = source_port
        self.target_port = target_port
        
        # Setup appearance
        pen = QPen(QColor(0, 0, 0), 2)
        self.setPen(pen)
        
        # Register connection with ports
        source_port.connections.append(self)
        target_port.connections.append(self)
        
        self.update_path()
    
    def update_path(self):
        """Update the line path based on port positions."""
        src = self.source_port.get_center()
        tgt = self.target_port.get_center()
        self.setLine(src.x(), src.y(), tgt.x(), tgt.y())
    
    def remove(self):
        """Remove this connection."""
        if self in self.source_port.connections:
            self.source_port.connections.remove(self)
        if self in self.target_port.connections:
            self.target_port.connections.remove(self)


class DSPGraph(QGraphicsScene):
    """Scene containing DSP blocks and connections."""
    
    block_added = pyqtSignal(DSPBlock)
    block_removed = pyqtSignal(DSPBlock)
    connection_added = pyqtSignal(Connection)
    connection_removed = pyqtSignal(Connection)
    
    def __init__(self):
        """Initialize DSP graph."""
        super().__init__()
        self.blocks: Dict[str, DSPBlock] = {}
        self.connections: List[Connection] = []
        self.counter = 0
        
        # Setup scene
        self.setSceneRect(-500, -500, 1000, 1000)
        self.setBackgroundBrush(QBrush(QColor(240, 240, 240)))
    
    def add_block(self, config: BlockConfig) -> DSPBlock:
        """Add a new block to the graph.
        
        Args:
            config: Block configuration
            
        Returns:
            Created DSP block
        """
        block = DSPBlock(config)
        self.addItem(block)
        self.blocks[config.id] = block
        self.block_added.emit(block)
        logger.info(f"Added block: {config.id} ({config.name})")
        return block
    
    def remove_block(self, block_id: str):
        """Remove a block from the graph.
        
        Args:
            block_id: ID of block to remove
        """
        if block_id not in self.blocks:
            return
        
        block = self.blocks[block_id]
        
        # Remove all connections
        for port in block.ports.values():
            for connection in list(port.connections):
                self.remove_connection(connection)
        
        self.removeItem(block)
        del self.blocks[block_id]
        self.block_removed.emit(block)
        logger.info(f"Removed block: {block_id}")
    
    def add_connection(self, source_port: Port, target_port: Port) -> Optional[Connection]:
        """Add a connection between two ports.
        
        Args:
            source_port: Source port
            target_port: Target port
            
        Returns:
            Created connection or None if invalid
        """
        # Validate connection
        if source_port.config.type == target_port.config.type:
            logger.warning("Cannot connect ports of same type")
            return None
        
        if source_port.block == target_port.block:
            logger.warning("Cannot connect ports on same block")
            return None
        
        # Create connection
        connection = Connection(source_port, target_port)
        self.addItem(connection)
        self.connections.append(connection)
        self.connection_added.emit(connection)
        logger.info(f"Added connection: {source_port.block.config.id} -> {target_port.block.config.id}")
        return connection
    
    def remove_connection(self, connection: Connection):
        """Remove a connection.
        
        Args:
            connection: Connection to remove
        """
        connection.remove()
        self.removeItem(connection)
        if connection in self.connections:
            self.connections.remove(connection)
        self.connection_removed.emit(connection)
        logger.info("Removed connection")
    
    def get_block_config(self, block_id: str) -> Optional[BlockConfig]:
        """Get configuration of a block.
        
        Args:
            block_id: ID of block
            
        Returns:
            Block configuration or None
        """
        if block_id in self.blocks:
            return self.blocks[block_id].config
        return None
    
    def to_dict(self) -> dict:
        """Serialize graph to dictionary.
        
        Returns:
            Serialized graph configuration
        """
        blocks = [block.config.to_dict() for block in self.blocks.values()]
        
        connections = []
        for conn in self.connections:
            connections.append({
                'source_block': conn.source_port.block.config.id,
                'source_port': conn.source_port.config.name,
                'target_block': conn.target_port.block.config.id,
                'target_port': conn.target_port.config.name
            })
        
        return {'blocks': blocks, 'connections': connections}


class BlockPaletteWidget(QListWidget):
    """Palette of available DSP blocks."""
    
    BLOCKS = {
        'Signal Source': {
            'type': BlockType.SOURCE,
            'ports': [
                PortConfig('out', PortType.OUTPUT, 'complex')
            ]
        },
        'FIR Filter': {
            'type': BlockType.FILTER,
            'ports': [
                PortConfig('in', PortType.INPUT, 'complex'),
                PortConfig('out', PortType.OUTPUT, 'complex')
            ]
        },
        'IIR Filter': {
            'type': BlockType.FILTER,
            'ports': [
                PortConfig('in', PortType.INPUT, 'complex'),
                PortConfig('out', PortType.OUTPUT, 'complex')
            ]
        },
        'QPSK Modulator': {
            'type': BlockType.MODULATOR,
            'ports': [
                PortConfig('data', PortType.INPUT, 'int'),
                PortConfig('out', PortType.OUTPUT, 'complex')
            ]
        },
        'QPSK Demodulator': {
            'type': BlockType.DEMODULATOR,
            'ports': [
                PortConfig('in', PortType.INPUT, 'complex'),
                PortConfig('out', PortType.OUTPUT, 'int')
            ]
        },
        'Energy Detector': {
            'type': BlockType.DETECTOR,
            'ports': [
                PortConfig('in', PortType.INPUT, 'complex'),
                PortConfig('out', PortType.OUTPUT, 'float')
            ]
        },
        'File Sink': {
            'type': BlockType.SINK,
            'ports': [
                PortConfig('in', PortType.INPUT, 'complex')
            ]
        },
        'Display Sink': {
            'type': BlockType.SINK,
            'ports': [
                PortConfig('in', PortType.INPUT, 'complex')
            ]
        }
    }
    
    def __init__(self):
        """Initialize block palette."""
        super().__init__()
        self.setDragDropMode(self.DragDropMode.DragOnly)
        self.populate_blocks()
    
    def populate_blocks(self):
        """Populate the palette with available blocks."""
        for block_name in self.BLOCKS.keys():
            item = QListWidgetItem(block_name)
            item.setData(Qt.ItemDataRole.UserRole, block_name)
            self.addItem(item)


class FlowgraphEditor(QMainWindow):
    """Main flowgraph editor window."""
    
    def __init__(self):
        """Initialize editor."""
        super().__init__()
        self.setWindowTitle("RF DSP Flowgraph Editor")
        self.setGeometry(100, 100, 1200, 800)
        
        # Create graph
        self.graph = DSPGraph()
        
        # Create canvas
        self.canvas = QGraphicsView(self.graph)
        self.canvas.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setCentralWidget(self.canvas)
        
        # Create block palette dock
        self.palette_widget = BlockPaletteWidget()
        palette_dock = QDockWidget("Block Palette", self)
        palette_dock.setWidget(self.palette_widget)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, palette_dock)
        
        # Create properties dock
        self.properties_widget = QWidget()
        self.properties_layout = QVBoxLayout()
        self.properties_widget.setLayout(self.properties_layout)
        properties_dock = QDockWidget("Properties", self)
        properties_dock.setWidget(self.properties_widget)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, properties_dock)
        
        # Setup menu bar
        self.setup_menus()
        
        # Setup status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Connect signals
        self.graph.block_added.connect(self.on_block_added)
        self.graph.block_removed.connect(self.on_block_removed)
        
        # Counter for unique IDs
        self.block_counter = 0
    
    def setup_menus(self):
        """Setup menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        file_menu.addAction("New", self.new_graph)
        file_menu.addAction("Open", self.open_graph)
        file_menu.addAction("Save", self.save_graph)
        file_menu.addSeparator()
        file_menu.addAction("Export", self.export_graph)
        file_menu.addSeparator()
        file_menu.addAction("Exit", self.close)
        
        # Edit menu
        edit_menu = menubar.addMenu("Edit")
        edit_menu.addAction("Clear All", self.clear_graph)
        edit_menu.addAction("Delete Selected", self.delete_selected)
        
        # View menu
        view_menu = menubar.addMenu("View")
        view_menu.addAction("Fit to View", self.fit_view)
        view_menu.addAction("Reset View", self.reset_view)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        help_menu.addAction("About", self.show_about)
    
    def new_graph(self):
        """Create new graph."""
        self.graph = DSPGraph()
        self.canvas.setScene(self.graph)
        self.block_counter = 0
        self.status_bar.showMessage("New graph created")
    
    def open_graph(self):
        """Open graph from file."""
        filepath, _ = QFileDialog.getOpenFileName(self, "Open Graph", "", "JSON Files (*.json)")
        if filepath:
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                self.graph = DSPGraph()
                self.canvas.setScene(self.graph)
                self.load_graph_from_dict(data)
                self.status_bar.showMessage(f"Loaded: {filepath}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load graph: {e}")
    
    def save_graph(self):
        """Save graph to file."""
        filepath, _ = QFileDialog.getSaveFileName(self, "Save Graph", "", "JSON Files (*.json)")
        if filepath:
            try:
                with open(filepath, 'w') as f:
                    json.dump(self.graph.to_dict(), f, indent=2)
                self.status_bar.showMessage(f"Saved: {filepath}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save graph: {e}")
    
    def export_graph(self):
        """Export graph configuration."""
        filepath, _ = QFileDialog.getSaveFileName(self, "Export Graph", "", "JSON Files (*.json)")
        if filepath:
            try:
                with open(filepath, 'w') as f:
                    json.dump(self.graph.to_dict(), f, indent=2)
                self.status_bar.showMessage(f"Exported: {filepath}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export graph: {e}")
    
    def clear_graph(self):
        """Clear all blocks from graph."""
        reply = QMessageBox.question(self, "Clear Graph", "Clear all blocks?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            block_ids = list(self.graph.blocks.keys())
            for block_id in block_ids:
                self.graph.remove_block(block_id)
            self.status_bar.showMessage("Graph cleared")
    
    def delete_selected(self):
        """Delete selected blocks."""
        for block in self.graph.blocks.values():
            if block.isSelected():
                self.graph.remove_block(block.config.id)
        self.status_bar.showMessage("Selected blocks deleted")
    
    def fit_view(self):
        """Fit view to all blocks."""
        self.canvas.fitInView(self.graph.itemsBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio)
    
    def reset_view(self):
        """Reset view to default."""
        self.canvas.resetTransform()
    
    def load_graph_from_dict(self, data: dict):
        """Load graph from dictionary.
        
        Args:
            data: Dictionary containing graph configuration
        """
        # Create blocks
        block_map = {}
        for block_data in data.get('blocks', []):
            block_type = BlockType(block_data['block_type'])
            ports = [
                PortConfig(
                    p['name'],
                    PortType(p['type']),
                    p.get('data_type', 'complex')
                )
                for p in block_data.get('ports', [])
            ]
            config = BlockConfig(
                block_data['id'],
                block_data['name'],
                block_type,
                block_data.get('x', 0),
                block_data.get('y', 0),
                block_data.get('params', {}),
                ports
            )
            block = self.graph.add_block(config)
            block_map[block_data['id']] = block
        
        # Create connections
        for conn_data in data.get('connections', []):
            src_block = block_map.get(conn_data['source_block'])
            tgt_block = block_map.get(conn_data['target_block'])
            
            if src_block and tgt_block:
                src_port = src_block.ports.get(conn_data['source_port'])
                tgt_port = tgt_block.ports.get(conn_data['target_port'])
                
                if src_port and tgt_port:
                    self.graph.add_connection(src_port, tgt_port)
    
    def mousePressEvent(self, event):
        """Handle mouse press for adding blocks."""
        super().mousePressEvent(event)
    
    def dragEnterEvent(self, event):
        """Handle drag enter."""
        if event.mimeData().hasFormat('application/x-qabstractitemmodeldatalist'):
            event.acceptProposedAction()
    
    def dropEvent(self, event):
        """Handle drop to add block."""
        item = self.palette_widget.itemFromIndex(self.palette_widget.indexAt(event.pos()))
        if item:
            block_name = item.data(Qt.ItemDataRole.UserRole)
            block_info = BlockPaletteWidget.BLOCKS[block_name]
            
            # Calculate drop position in scene coordinates
            canvas_pos = self.canvas.mapFromGlobal(event.globalPos())
            scene_pos = self.canvas.mapToScene(canvas_pos)
            
            # Create block config
            config = BlockConfig(
                f"block_{self.block_counter}",
                block_name,
                block_info['type'],
                scene_pos.x(),
                scene_pos.y(),
                {},
                block_info['ports']
            )
            
            self.graph.add_block(config)
            self.block_counter += 1
            self.status_bar.showMessage(f"Added {block_name}")
            event.acceptProposedAction()
    
    def on_block_added(self, block: DSPBlock):
        """Handle block added event.
        
        Args:
            block: Added DSP block
        """
        self.status_bar.showMessage(f"Added block: {block.config.name}")
    
    def on_block_removed(self, block: DSPBlock):
        """Handle block removed event.
        
        Args:
            block: Removed DSP block
        """
        self.status_bar.showMessage(f"Removed block: {block.config.name}")
    
    def show_about(self):
        """Show about dialog."""
        QMessageBox.information(self, "About", 
                               "RF DSP Flowgraph Editor\n\n"
                               "A visual editor for designing RF signal processing pipelines.\n\n"
                               "Drag blocks from the palette to the canvas and connect them.")


def main():
    """Main entry point."""
    app = QApplication(sys.argv)
    editor = FlowgraphEditor()
    editor.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
