// Create a comment with summary PDF, par, and tim files
// and update it as new commits come in.
// This script was originally written by Claude Opus 5.5.
// Some expanatory comments added by RJJ.

// The action places this marker in the comment it creates.
// It is used later to find the comment so it can be updated.
const MARKER = '<!-- notebook-artifacts -->';

// This function becomes the full contents of the (CommonJS) module,
// and is invoked when the module is loaded in commenting_artifacts.yml.
module.exports = async ({ github, context, core }) => {
  // The `github`, `context`, and `core` objects are provided by the github-script
  // action (`https://github.com/actions/github-script`), as noted in its README.
  //
  // The `github` object is an authenticated octokit client that provides access
  // to the GitHub REST API through github.rest.* methods, as well as some other
  // methods for things like pagination. Complete documentation for this can be
  // found at `https://octokit.github.io/rest.js`. Examples there use the name
  // `octokit` rather than `github` for the object.
  // 
  // The `context` object contains the values of various environment variables
  // set by GitHub for the action being run. These variables are documented at
  // `https://docs.github.com/en/actions/reference/workflows-and-actions/variables`,
  // and the mapping is mostly obvious (e.g., GITHUB_REPO becomes context.repo),
  // but can also be easily read from a section of code linked to from the README.
  // Additionally, it contains the webhook payload corresponding to the event that
  // was run, as `context.payload`. For this script, this is always `workflow_run`.
  // The parameters in the webhook payload are documented at
  // `https://docs.github.com/en/webhooks/webhook-events-and-payloads`.
  //
  // The `core` object is provided by the `core` action, found in the repo at
  // `https://github.com/actions/core`. Here, it is mainly used for logging.
  // Its various methods are documented in the README for the `core` action.

  const run = context.payload.workflow_run;
  const { owner, repo } = context.repo;

  // pull_requests is empty for fork PRs, so fall back to a head-branch lookup
  let prNumber = run.pull_requests?.[0]?.number;
  if (!prNumber) {
    const { data: prs } = await github.rest.pulls.list({
      owner, repo, state: 'open',
      head: `${run.head_repository.owner.login}:${run.head_branch}`,
    });
    prNumber = prs.find(p => p.head.sha === run.head_sha)?.number;
  }
  if (!prNumber) {
    core.setFailed(`Could not find a PR for commit ${run.head_sha}`);
    return;
  }

  const artifacts = await github.paginate(
    github.rest.actions.listWorkflowRunArtifacts,
    { owner, repo, run_id: run.id },
  );
  if (artifacts.length === 0) {
    core.info('No artifacts found; leaving any existing comment unchanged.');
    return;
  }

  artifacts.sort((a, b) => a.name.localeCompare(b.name));
  const rows = artifacts.map(a => {
    const url = `https://github.com/${owner}/${repo}/actions/runs/${run.id}/artifacts/${a.id}`;
    const label = a.name.endsWith('.pdf') ? 'View PDF' : 'Download ZIP';
    return `| ${a.name} | [${label}](${url}) |`;
  });

  const body = [
    MARKER,
    '### Notebook pipeline artifacts',
    `From [run #${run.run_number}](${run.html_url}) on commit \`${run.head_sha.slice(0, 7)}\`.`,
    '',
    '| Artifact | Link |',
    '|----------|------|',
    ...rows,
  ].join('\n');

  const comments = await github.paginate(
    github.rest.issues.listComments,
    { owner, repo, issue_number: prNumber, per_page: 100 },
  );
  const existing = comments.find(
    c => c.user?.login === 'github-actions[bot]' && c.body?.includes(MARKER),
  );

  if (existing) {
    await github.rest.issues.updateComment({ owner, repo, comment_id: existing.id, body });
    core.info(`Updated comment ${existing.id} on PR #${prNumber}`);
  } else {
    await github.rest.issues.createComment({ owner, repo, issue_number: prNumber, body });
    core.info(`Created comment on PR #${prNumber}`);
  }
};
