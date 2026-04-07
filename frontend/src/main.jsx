import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'
import ScrollViz from './ScrollViz.jsx'
import ExploreViz from './ExploreViz.jsx'
import { ThemeProvider } from './useTheme.jsx'

// Use query params to show different views
const params = window.location.search;
const View = params.includes('scroll') ? ScrollViz
           : params.includes('explore') ? ExploreViz
           : App;

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <ThemeProvider>
      <View />
    </ThemeProvider>
  </StrictMode>,
)
