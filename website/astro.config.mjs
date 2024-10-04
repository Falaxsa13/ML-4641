// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

// https://astro.build/config
export default defineConfig({
	base: '/ML-4641/',
	integrations: [
		starlight({
			title: 'My Docs',
			social: {
				github: 'https://github.com/withastro/starlight',
			},
			sidebar: [
				{
					label: 'Proposal',
					items: [
						// Each item here is one entry in the navigation menu.
						{ label: 'Introduction', slug: 'proposal/introduction' },
						{ label: 'Problem Definition', slug: 'proposal/problem' },
						{ label: 'Methodology', slug: 'proposal/methodology' },
						{ label: 'Results and Discussion ', slug: 'proposal/result' },
					],
				},
				{
					label: 'Reference',
					autogenerate: { directory: 'reference' },
				},
			],
		}),
	],
});
