import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import ExploreViz from "./ExploreViz";

// Standalone entry point for the ExploreViz prototype.
// Can be used as an alternative main entry or rendered alongside the main app.
//
// To use as the Vite entry point, create an explore.html in the project root:
//   <div id="explore-root"></div>
//   <script type="module" src="/src/ExploreEntry.jsx"></script>
//
// Or import ExploreViz directly into App.jsx as a tab/route.

const root = document.getElementById("explore-root") || document.getElementById("root");

if (root) {
  createRoot(root).render(
    <StrictMode>
      <ExploreViz />
    </StrictMode>
  );
}

export default ExploreViz;
