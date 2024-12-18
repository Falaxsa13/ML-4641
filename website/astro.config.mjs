// @ts-check
import { defineConfig } from "astro/config";
import starlight from "@astrojs/starlight";

// https://astro.build/config
export default defineConfig({
  base: "/ML-4641/",
  integrations: [
    starlight({
      title: "HeartGuard AI",
      customCss: [
        "./src/styles/custom.css", // Path to your custom CSS
      ],
      social: {
        github: "https://github.com/withastro/starlight",
      },
      sidebar: [
        {
          label: "Proposal",
          items: [
            // Each item here is one entry in the navigation menu.
            { label: "Introduction", slug: "proposal/introduction" },
            { label: "Problem Definition", slug: "proposal/problem" },
            { label: "Methodology", slug: "proposal/methodology" },
            { label: "Results and Discussion ", slug: "proposal/result" },
            { label: "References", slug: "proposal/references" },
          ],
        },
        {
          label: "Midterm",
          items: [
            // Each item here is one entry in the navigation menu.
            { label: "Introduction", slug: "midterm/introduction" },
            { label: "Problem Definition", slug: "midterm/problem" },
            { label: "Methods", slug: "midterm/methods" },
            { label: "Results and Discussion ", slug: "midterm/results" },
            { label: "References", slug: "midterm/references" },
          ],
        },
        {
          label: "Final",
          items: [
            // Each item here is one entry in the navigation menu.
            { label: "Introduction", slug: "final/introduction" },
            { label: "Problem Definition", slug: "final/problem" },
            { label: "Methods", slug: "final/methods" },
            { label: "Results and Discussion ", slug: "final/results" },
            { label: "References", slug: "final/references" },
          ],
        },
      ],
    }),
  ],
});
