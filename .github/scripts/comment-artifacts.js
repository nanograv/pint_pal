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
  // and the mapping is mostly obvious (e.g., GITHUB_SHA becomes context.sha),
  // but can also be read from a section of code linked to from the README.
  // Additionally, it contains the webhook payload corresponding to the event that
  // was run (here, always `workflow_run`), as `context.payload`.
  // The parameters in the webhook payload are documented at
  // `https://docs.github.com/en/webhooks/webhook-events-and-payloads`.
  //
  // The `core` object is provided by the `core` action, found in the repo at
  // `https://github.com/actions/core`. Here, it is mainly used for logging.
  // Its various methods are documented in the README for the `core` action.

  // Unpack relevant pieces of context.
  // We can assume the event type is `workflow_run` because that is the way the
  // trigger is set up in `commenting_artifacts.yml`.
  const run = context.payload.workflow_run;
  // The value of `context.repo` is the owner and name of the current repo, e.g.
  // `{ "owner": "nanograv", "repo": "pint_pal" }`. In a fork, it will point to
  // the fork instead. This is based on the GITHUB_REPOSITORY env variable, but
  // that would instead have a single string value like "nanograv/pint_pal".
  const { owner, repo } = context.repo;

  // Determine the PR number, so we can find it to post the comment.
  // For PRs from a branch in the same repo we can get this from
  // `run.pull_requests`, but this is empty for PRs that came from a fork
  // (see `https://github.com/github/docs/issues/22501`),
  // so in that case we fall back to listing the PRs and finding the one with
  // the same head branch.
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

  // Find the artifacts produced by this run.
  const artifacts = await github.paginate(
    github.rest.actions.listWorkflowRunArtifacts,
    { owner, repo, run_id: run.id },
  );
  if (artifacts.length === 0) {
    core.info('No artifacts found; leaving any existing comment unchanged.');
    return;
  }

  // Build up the rows of the markdown table, with a link to each artifact.
  // Artifacts are sorted alphabetically using `localeCompare()`.
  // For PDF artifacts, we provide a direct link to view the PDF.
  artifacts.sort((a, b) => a.name.localeCompare(b.name));
  const rows = artifacts.map(a => {
    const url = `https://github.com/${owner}/${repo}/actions/runs/${run.id}/artifacts/${a.id}`;
    const label = a.name.endsWith('.pdf') ? 'View PDF' : 'Download ZIP';
    return `| ${a.name} | [${label}](${url}) |`;
  });

  // Construct the body of the comment using the marker and table rows.
  const body = [
    MARKER,
    '### Notebook pipeline artifacts',
    `From [run #${run.run_number}](${run.html_url}) on commit \`${run.head_sha.slice(0, 7)}\`.`,
    '',
    '| Artifact | Link |',
    '|----------|------|',
    ...rows,
  ].join('\n');

  // List the existing comments and go through them to find if there is already
  // a comment containing the hidden marker text.
  const comments = await github.paginate(
    github.rest.issues.listComments,
    { owner, repo, issue_number: prNumber, per_page: 100 },
  );
  const existing = comments.find(
    c => c.user?.login === 'github-actions[bot]' && c.body?.includes(MARKER),
  );

  // Create a new comment using the body text, or, if one already exists, update it.
  if (existing) {
    await github.rest.issues.updateComment({ owner, repo, comment_id: existing.id, body });
    core.info(`Updated comment ${existing.id} on PR #${prNumber}`);
  } else {
    await github.rest.issues.createComment({ owner, repo, issue_number: prNumber, body });
    core.info(`Created comment on PR #${prNumber}`);
  }
};
