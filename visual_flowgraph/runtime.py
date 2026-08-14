"""Runtime binding of flowgraph to DSP blocks."""

import json
import numpy as np
from typing import Dict, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum
import logging
from queue import Queue

logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    """Execution modes for the flowgraph."""
    OFFLINE = "offline"      # Process all data at once
    STREAMING = "streaming"  # Process data in real-time


@dataclass
class ProcessingBlock:
    """Base class for processing blocks."""
    
    name: str
    block_id: str
    input_ports: Dict[str, 'PortBuffer']
    output_ports: Dict[str, 'PortBuffer']
    
    def process(self) -> bool:
        """Process data. Returns True if successful."""
        raise NotImplementedError
    
    def reset(self):
        """Reset block state."""
        pass


class PortBuffer:
    """Buffer for data flowing through a port."""
    
    def __init__(self, buffer_size: int = 1000):
        """Initialize port buffer.
        
        Args:
            buffer_size: Maximum buffer size
        """
        self.buffer: Queue = Queue(maxsize=buffer_size)
        self.buffer_size = buffer_size
    
    def put(self, data: np.ndarray):
        """Put data into buffer.
        
        Args:
            data: Data to put
        """
        self.buffer.put(data)
    
    def get(self) -> Optional[np.ndarray]:
        """Get data from buffer.
        
        Returns:
            Data or None if buffer is empty
        """
        try:
            return self.buffer.get_nowait()
        except:
            return None
    
    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        return self.buffer.empty()
    
    def size(self) -> int:
        """Get current buffer size."""
        return self.buffer.qsize()


class SignalSource(ProcessingBlock):
    """Signal source block."""
    
    def __init__(self, block_id: str, signal_data: np.ndarray, **kwargs):
        """Initialize signal source.
        
        Args:
            block_id: Block ID
            signal_data: Signal data to generate
        """
        super().__init__("Signal Source", block_id, {}, {"out": PortBuffer()})
        self.signal_data = signal_data
        self.index = 0
        self.chunk_size = kwargs.get('chunk_size', 256)
    
    def process(self) -> bool:
        """Generate signal data."""
        if self.index >= len(self.signal_data):
            return False  # No more data
        
        end_idx = min(self.index + self.chunk_size, len(self.signal_data))
        chunk = self.signal_data[self.index:end_idx]
        self.output_ports["out"].put(chunk)
        self.index = end_idx
        return True
    
    def reset(self):
        """Reset to beginning."""
        self.index = 0


class FIRFilter(ProcessingBlock):
    """FIR filter block."""
    
    def __init__(self, block_id: str, coefficients: np.ndarray, **kwargs):
        """Initialize FIR filter.
        
        Args:
            block_id: Block ID
            coefficients: Filter coefficients
        """
        super().__init__("FIR Filter", block_id,
                        {"in": PortBuffer()},
                        {"out": PortBuffer()})
        self.coefficients = coefficients
        self.state = np.zeros(len(coefficients) - 1, dtype=complex)
    
    def process(self) -> bool:
        """Apply FIR filtering."""
        data = self.input_ports["in"].get()
        if data is None:
            return False
        
        # Apply filter using scipy.signal.lfilter equivalent
        output = np.convolve(data, self.coefficients, mode='same')
        self.output_ports["out"].put(output)
        return True
    
    def reset(self):
        """Reset filter state."""
        self.state = np.zeros(len(self.coefficients) - 1, dtype=complex)


class EnergyDetector(ProcessingBlock):
    """Energy detector block."""
    
    def __init__(self, block_id: str, **kwargs):
        """Initialize energy detector.
        
        Args:
            block_id: Block ID
        """
        super().__init__("Energy Detector", block_id,
                        {"in": PortBuffer()},
                        {"out": PortBuffer()})
    
    def process(self) -> bool:
        """Compute energy."""
        data = self.input_ports["in"].get()
        if data is None:
            return False
        
        # Compute energy per sample
        energy = np.abs(data) ** 2
        self.output_ports["out"].put(energy)
        return True


class FileSink(ProcessingBlock):
    """File sink block for recording data."""
    
    def __init__(self, block_id: str, filepath: str, **kwargs):
        """Initialize file sink.
        
        Args:
            block_id: Block ID
            filepath: Path to output file
        """
        super().__init__("File Sink", block_id,
                        {"in": PortBuffer()},
                        {})
        self.filepath = filepath
        self.data = []
    
    def process(self) -> bool:
        """Write data to file."""
        data = self.input_ports["in"].get()
        if data is None:
            return False
        
        self.data.append(data)
        return True
    
    def finalize(self):
        """Save accumulated data to file."""
        if self.data:
            combined = np.concatenate(self.data)
            np.save(self.filepath, combined)
            logger.info(f"Saved data to {self.filepath}")


class FlowgraphRuntime:
    """Runtime for executing configured flowgraphs."""
    
    def __init__(self, mode: ExecutionMode = ExecutionMode.OFFLINE):
        """Initialize flowgraph runtime.
        
        Args:
            mode: Execution mode
        """
        self.mode = mode
        self.blocks: Dict[str, ProcessingBlock] = {}
        self.connections: Dict[str, list] = {}  # source_port -> [target_ports]
        self.execution_order: list = []
        self.is_running = False
    
    def add_block(self, block: ProcessingBlock):
        """Add a processing block.
        
        Args:
            block: Processing block to add
        """
        self.blocks[block.block_id] = block
        logger.info(f"Added block: {block.block_id} ({block.name})")
    
    def add_connection(self, source_id: str, source_port: str,
                      target_id: str, target_port: str):
        """Add a connection between blocks.
        
        Args:
            source_id: ID of source block
            source_port: Name of source port
            target_id: ID of target block
            target_port: Name of target port
        """
        source_key = f"{source_id}:{source_port}"
        
        if source_key not in self.connections:
            self.connections[source_key] = []
        
        target_key = f"{target_id}:{target_port}"
        self.connections[source_key].append((target_id, target_port))
        logger.info(f"Connected {source_key} -> {target_key}")
    
    def load_from_config(self, config: dict, block_factory: Dict[str, Callable]):
        """Load flowgraph from configuration dictionary.
        
        Args:
            config: Configuration dictionary
            block_factory: Dict mapping block types to factory functions
        """
        # Create blocks
        for block_data in config.get('blocks', []):
            block_type = block_data['block_type']
            block_id = block_data['id']
            params = block_data.get('params', {})
            
            if block_type in block_factory:
                block = block_factory[block_type](block_id, **params)
                self.add_block(block)
            else:
                logger.warning(f"Unknown block type: {block_type}")
        
        # Create connections
        for conn_data in config.get('connections', []):
            self.add_connection(
                conn_data['source_block'],
                conn_data['source_port'],
                conn_data['target_block'],
                conn_data['target_port']
            )
    
    def propagate_data(self):
        """Propagate data through connections."""
        for source_key, targets in self.connections.items():
            source_id, source_port = source_key.split(':')
            source_block = self.blocks[source_id]
            
            # Get data from source
            if source_port in source_block.output_ports:
                data = source_block.output_ports[source_port].get()
                
                if data is not None:
                    # Send to all targets
                    for target_id, target_port in targets:
                        target_block = self.blocks[target_id]
                        if target_port in target_block.input_ports:
                            target_block.input_ports[target_port].put(data)
    
    def execute_step(self) -> bool:
        """Execute one processing step.
        
        Returns:
            True if any block produced output, False otherwise
        """
        any_output = False
        
        for block in self.blocks.values():
            if block.process():
                any_output = True
        
        self.propagate_data()
        return any_output
    
    def execute(self, max_iterations: Optional[int] = None) -> Dict[str, Any]:
        """Execute the entire flowgraph.
        
        Args:
            max_iterations: Maximum iterations (None for unlimited)
            
        Returns:
            Execution statistics
        """
        logger.info("Starting flowgraph execution")
        self.is_running = True
        
        # Reset all blocks
        for block in self.blocks.values():
            block.reset()
        
        iteration = 0
        while (max_iterations is None or iteration < max_iterations) and self.is_running:
            if not self.execute_step():
                break
            iteration += 1
        
        # Finalize sinks
        for block in self.blocks.values():
            if isinstance(block, FileSink):
                block.finalize()
        
        self.is_running = False
        logger.info(f"Flowgraph execution complete. Iterations: {iteration}")
        
        return {'iterations': iteration, 'blocks': len(self.blocks)}
    
    def stop(self):
        """Stop execution."""
        self.is_running = False
    
    def reset(self):
        """Reset all blocks."""
        for block in self.blocks.values():
            block.reset()


# Block factory functions
def create_signal_source(block_id: str, signal_data: Optional[np.ndarray] = None, **kwargs) -> SignalSource:
    """Factory for signal source blocks."""
    if signal_data is None:
        signal_data = np.random.randn(1000) + 1j * np.random.randn(1000)
    return SignalSource(block_id, signal_data, **kwargs)


def create_fir_filter(block_id: str, coefficients: Optional[np.ndarray] = None, **kwargs) -> FIRFilter:
    """Factory for FIR filter blocks."""
    if coefficients is None:
        # Default: simple moving average
        coefficients = np.ones(5) / 5
    return FIRFilter(block_id, coefficients, **kwargs)


def create_energy_detector(block_id: str, **kwargs) -> EnergyDetector:
    """Factory for energy detector blocks."""
    return EnergyDetector(block_id, **kwargs)


def create_file_sink(block_id: str, filepath: str = "output.npy", **kwargs) -> FileSink:
    """Factory for file sink blocks."""
    return FileSink(block_id, filepath, **kwargs)


# Default block factory mapping
DEFAULT_BLOCK_FACTORY = {
    'source': create_signal_source,
    'filter': create_fir_filter,
    'detector': create_energy_detector,
    'sink': create_file_sink
}
