// Create a comment with summary PDF, par, and tim files
// and update it as new commits come in

const MARKER = '<!-- notebook-artifacts -->';

module.exports = async ({ github, context, core }) => {
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
