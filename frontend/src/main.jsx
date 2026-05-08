import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'
import VizPrototypes from './VizPrototypes.jsx'

const params = window.location.search;
const View = params.includes('prototypes') ? VizPrototypes : App;

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <View />
  </StrictMode>,
)
