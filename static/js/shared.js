// Shared JavaScript functions for T2 Map Demo


// Convert 2D array to image data URL
function arrayToImageSrc(array2d, colormap = 'gray', minVal = null, maxVal = null) {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    
    const height = array2d.length;
    const width = array2d[0].length;
    canvas.width = width;
    canvas.height = height;
    
    const imageData = ctx.createImageData(width, height);
    
    // Find min/max for normalization
    let min = minVal !== null ? minVal : Infinity;
    let max = maxVal !== null ? maxVal : -Infinity;
    
    if (minVal === null || maxVal === null) {
        // Auto-calculate min/max
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                min = Math.min(min, array2d[y][x]);
                max = Math.max(max, array2d[y][x]);
            }
        }
    }
    
    // Apply colormap
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const normalized = (array2d[y][x] - min) / (max - min);
            const index = (y * width + x) * 4;
            
            if (colormap === 'viridis') {
                // Map to viridis colormap
                const colorIndex = Math.floor(normalized * 255);
                const clampedIndex = Math.max(0, Math.min(255, colorIndex));
                const [r, g, b] = viridisColors[clampedIndex];
                imageData.data[index] = r;
                imageData.data[index + 1] = g;
                imageData.data[index + 2] = b;
                imageData.data[index + 3] = 255;
            } else {
                // Default grayscale
                const grayValue = normalized * 255;
                imageData.data[index] = grayValue;
                imageData.data[index + 1] = grayValue;
                imageData.data[index + 2] = grayValue;
                imageData.data[index + 3] = 255;
            }
        }
    }
    
    ctx.putImageData(imageData, 0, 0);
    return canvas.toDataURL('image/png');
}

// Viridis colormap lookup table (256 entries)
const viridisColors = [
  [68,1,84],[68,2,86],[69,4,87],[69,5,89],[70,7,90],[70,8,92],[70,10,93],[70,11,94],[71,13,96],
  [71,14,97],[71,16,99],[71,17,100],[71,19,101],[72,20,103],[72,22,104],[72,23,105],[72,24,106],
  [72,26,108],[72,27,109],[72,28,110],[72,29,111],[72,31,112],[72,32,113],[72,33,115],[72,35,116],
  [72,36,117],[72,37,118],[72,38,119],[72,40,120],[72,41,121],[71,42,122],[71,44,122],[71,45,123],
  [71,46,124],[71,47,125],[70,48,126],[70,50,126],[70,51,127],[70,52,128],[69,53,129],[69,55,129],
  [69,56,130],[68,57,131],[68,58,131],[68,59,132],[67,61,132],[67,62,133],[66,63,133],[66,64,134],
  [66,65,134],[65,66,135],[65,68,135],[64,69,136],[64,70,136],[63,71,136],[63,72,137],[62,73,137],
  [62,74,137],[62,76,138],[61,77,138],[61,78,138],[60,79,138],[60,80,139],[59,81,139],[59,82,139],
  [58,83,139],[58,84,140],[57,85,140],[57,86,140],[56,88,140],[56,89,140],[55,90,140],[55,91,141],
  [54,92,141],[54,93,141],[53,94,141],[53,95,141],[52,96,141],[52,97,141],[51,98,141],[51,99,141],
  [50,100,142],[50,101,142],[49,102,142],[49,103,142],[49,104,142],[48,105,142],[48,106,142],
  [47,107,142],[47,108,142],[46,109,142],[46,110,142],[46,111,142],[45,112,142],[45,113,142],
  [44,113,142],[44,114,142],[44,115,142],[43,116,142],[43,117,142],[42,118,142],[42,119,142],
  [42,120,142],[41,121,142],[41,122,142],[41,123,142],[40,124,142],[40,125,142],[39,126,142],
  [39,127,142],[39,128,142],[38,129,142],[38,130,142],[38,130,142],[37,131,142],[37,132,142],
  [37,133,142],[36,134,142],[36,135,142],[35,136,142],[35,137,142],[35,138,141],[34,139,141],
  [34,140,141],[34,141,141],[33,142,141],[33,143,141],[33,144,141],[33,145,140],[32,146,140],
  [32,146,140],[32,147,140],[31,148,140],[31,149,139],[31,150,139],[31,151,139],[31,152,139],
  [31,153,138],[31,154,138],[30,155,138],[30,156,137],[30,157,137],[31,158,137],[31,159,136],
  [31,160,136],[31,161,136],[31,161,135],[31,162,135],[32,163,134],[32,164,134],[33,165,133],
  [33,166,133],[34,167,133],[34,168,132],[35,169,131],[36,170,131],[37,171,130],[37,172,130],
  [38,173,129],[39,173,129],[40,174,128],[41,175,127],[42,176,127],[44,177,126],[45,178,125],
  [46,179,124],[47,180,124],[49,181,123],[50,182,122],[52,182,121],[53,183,121],[55,184,120],
  [56,185,119],[58,186,118],[59,187,117],[61,188,116],[63,188,115],[64,189,114],[66,190,113],
  [68,191,112],[70,192,111],[72,193,110],[74,193,109],[76,194,108],[78,195,107],[80,196,106],
  [82,197,105],[84,197,104],[86,198,103],[88,199,101],[90,200,100],[92,200,99],[94,201,98],
  [96,202,96],[99,203,95],[101,203,94],[103,204,92],[105,205,91],[108,205,90],[110,206,88],
  [112,207,87],[115,208,86],[117,208,84],[119,209,83],[122,209,81],[124,210,80],[127,211,78],
  [129,211,77],[132,212,75],[134,213,73],[137,213,72],[139,214,70],[142,214,69],[144,215,67],
  [147,215,65],[149,216,64],[152,216,62],[155,217,60],[157,217,59],[160,218,57],[162,218,55],
  [165,219,54],[168,219,52],[170,220,50],[173,220,48],[176,221,47],[178,221,45],[181,222,43],
  [184,222,41],[186,222,40],[189,223,38],[192,223,37],[194,223,35],[197,224,33],[200,224,32],
  [202,225,31],[205,225,29],[208,225,28],[210,226,27],[213,226,26],[216,226,25],[218,227,25],
  [221,227,24],[223,227,24],[226,228,24],[229,228,25],[231,228,25],[234,229,26],[236,229,27],
  [239,229,28],[241,229,29],[244,230,30],[246,230,32],[248,230,33],[251,231,35],[253,231,37]
];

// =============================================================================
// Generic Zoom/Pan Functionality
// =============================================================================

// Global state for zoom/pan functionality
const zoomPanGlobalState = {
  syncStates: {},           // Stores shared state for synchronized zoom groups
  activeContainer: null,    // Currently active container being dragged
  handlersInitialized: false
};

/**
 * Initialize zoom/pan for a container with a specific transform application function
 * @param {HTMLElement} container - The container to enable zoom/pan on
 * @param {string} syncKey - Optional key for synchronizing multiple containers
 * @param {Function} applyTransformFn - Function to apply transform to this container
 * @param {Function} applyGroupFn - Function to apply transform to all containers in sync group
 */
function initZoomPanForContainer(container, syncKey, applyTransformFn, applyGroupFn) {
  if (!container) {
    return;
  }
  
  ensureZoomPanHandlers();
  
  // Initialize sync state if needed
  if (syncKey && !zoomPanGlobalState.syncStates[syncKey]) {
    zoomPanGlobalState.syncStates[syncKey] = { scale: 1, translateX: 0, translateY: 0 };
  }
  
  const sharedState = syncKey ? zoomPanGlobalState.syncStates[syncKey] : { scale: 1, translateX: 0, translateY: 0 };
  
  container._zoomState = {
    syncKey: syncKey,
    sharedState: sharedState,
    isDragging: false,
    dragMode: null,
    lastX: 0,
    lastY: 0,
    applyTransform: applyTransformFn,
    applyGroup: applyGroupFn
  };
  
  applyTransformFn(container);
  container.style.cursor = 'crosshair';
  
  if (syncKey && applyGroupFn) {
    applyGroupFn(syncKey);
  }
  
  container.addEventListener('contextmenu', function(e) {
    e.preventDefault();
  });
  
  container.addEventListener('mousedown', function(e) {
    // Skip if clicking on excluded elements (handles, grips, etc)
    if (e.target.closest('.comparison-handle') || e.target.closest('.comparison-grip')) {
      return;
    }
    if (e.button !== 2 && e.button !== 1) {
      return;
    }
    e.preventDefault();
    const state = container._zoomState;
    state.isDragging = true;
    state.dragMode = (e.button === 2) ? 'zoom' : 'pan';
    state.lastX = e.clientX;
    state.lastY = e.clientY;
    container.style.cursor = state.dragMode === 'zoom' ? 'ns-resize' : 'move';
    zoomPanGlobalState.activeContainer = container;
  });
  
  container.addEventListener('mouseup', function() {
    if (zoomPanGlobalState.activeContainer === container) {
      handleZoomMouseUp();
    }
  });
}

/**
 * Ensure global zoom/pan event handlers are registered
 */
function ensureZoomPanHandlers() {
  if (zoomPanGlobalState.handlersInitialized) {
    return;
  }
  zoomPanGlobalState.handlersInitialized = true;
  document.addEventListener('mousemove', handleZoomMouseMove);
  document.addEventListener('mouseup', handleZoomMouseUp);
  document.addEventListener('dblclick', handleZoomDoubleClick);
}

/**
 * Handle mouse move for zoom/pan
 */
function handleZoomMouseMove(e) {
  if (!zoomPanGlobalState.activeContainer) {
    return;
  }
  const state = zoomPanGlobalState.activeContainer._zoomState;
  if (!state || !state.isDragging) {
    return;
  }
  
  const deltaX = e.clientX - state.lastX;
  const deltaY = e.clientY - state.lastY;
  const shared = state.sharedState;
  
  if (state.dragMode === 'zoom') {
    const zoomSensitivity = 0.01;
    const newScale = Math.max(0.5, Math.min(10, shared.scale - deltaY * zoomSensitivity));
    shared.scale = newScale;
  } else if (state.dragMode === 'pan') {
    shared.translateX += deltaX;
    shared.translateY += deltaY;
  }
  
  state.lastX = e.clientX;
  state.lastY = e.clientY;
  
  // Apply transform to synchronized group or single container
  if (state.syncKey && state.applyGroup) {
    state.applyGroup(state.syncKey);
  } else if (state.applyTransform) {
    state.applyTransform(zoomPanGlobalState.activeContainer);
  }
}

/**
 * Handle mouse up for zoom/pan
 */
function handleZoomMouseUp() {
  if (!zoomPanGlobalState.activeContainer) {
    return;
  }
  const state = zoomPanGlobalState.activeContainer._zoomState;
  if (state) {
    state.isDragging = false;
    state.dragMode = null;
  }
  zoomPanGlobalState.activeContainer.style.cursor = 'crosshair';
  zoomPanGlobalState.activeContainer = null;
}

/**
 * Handle double-click to reset zoom/pan
 */
function handleZoomDoubleClick(e) {
  const container = e.target.closest('[data-zoom-enabled]');
  if (!container || !container._zoomState) {
    return;
  }
  resetZoomForContainer(container);
}

/**
 * Reset zoom/pan for a container
 * @param {HTMLElement} container - The container to reset
 */
function resetZoomForContainer(container) {
  if (!container || !container._zoomState) {
    return;
  }
  const state = container._zoomState;
  const shared = state.sharedState;
  shared.scale = 1;
  shared.translateX = 0;
  shared.translateY = 0;
  
  // Apply reset to synchronized group or single container
  if (state.syncKey && state.applyGroup) {
    state.applyGroup(state.syncKey);
  } else if (state.applyTransform) {
    state.applyTransform(container);
  }
}

/**
 * Initialize zoom sync state for a group
 * @param {string} syncKey - The key identifying the sync group
 */
function initZoomSyncState(syncKey) {
  if (syncKey && !zoomPanGlobalState.syncStates[syncKey]) {
    zoomPanGlobalState.syncStates[syncKey] = { scale: 1, translateX: 0, translateY: 0 };
  }
}

/**
 * Reset active zoom interaction (useful when switching views/tabs)
 */
function resetActiveZoomInteraction() {
  if (zoomPanGlobalState.activeContainer) {
    const activeState = zoomPanGlobalState.activeContainer._zoomState;
    if (activeState) {
      activeState.isDragging = false;
      activeState.dragMode = null;
    }
    zoomPanGlobalState.activeContainer.style.cursor = 'crosshair';
    zoomPanGlobalState.activeContainer = null;
  }
}