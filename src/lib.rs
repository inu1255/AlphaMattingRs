#[macro_use]
extern crate vulkano;
extern crate image;
extern crate vulkano_shaders;

use std::path::Path;
use libc::c_char;
use std::ffi::CStr;
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBuffer, DynamicState};
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::device::{Device, DeviceExtensions};
use vulkano::format::Format;
use vulkano::framebuffer::{Framebuffer, Subpass};
use vulkano::image::{Dimensions, ImmutableImage};
use vulkano::instance::{Instance, PhysicalDevice};
use vulkano::pipeline::viewport::Viewport;
use vulkano::pipeline::GraphicsPipeline;
use vulkano::sampler::{Filter, MipmapMode, Sampler, SamplerAddressMode};
use vulkano::sync::GpuFuture;

use image::{ImageBuffer, Rgba};

use std::sync::Arc;
use std::collections::LinkedList;
use vulkano::device::Features;
use vulkano::instance::InstanceExtensions;
use vulkano::image::StorageImage;

mod quad_vs {
    vulkano_shaders::shader!{
        ty: "vertex",
        path: "shaders/quad.vert.glsl"
    }
}

mod copy_fs {
    vulkano_shaders::shader!{
        ty: "fragment",
        // path: "shaders/test.frag.glsl"
        path: "shaders/copy.frag.glsl"
    }
}

mod expand_known_fs {
    vulkano_shaders::shader!{
        ty: "fragment",
        // path: "shaders/test.frag.glsl"
        path: "shaders/expandKnown.frag.glsl"
    }
}

mod gathering_fs {
    vulkano_shaders::shader!{
        ty: "fragment",
        path: "shaders/gathering.frag.glsl"
    }
}

mod refinement_fs {
    vulkano_shaders::shader!{
        ty: "fragment",
        path: "shaders/refinement.frag.glsl"
    }
}

mod local_smooth_fs {
    vulkano_shaders::shader!{
        ty: "fragment",
        path: "shaders/localSmooth.frag.glsl"
    }
}

#[derive(Debug, Clone)]
struct Vertex {
    position: [f32; 2],
}
impl_vertex!(Vertex, position);

pub struct Shared {
    queue: Arc<vulkano::device::Queue>,
    device: Arc<vulkano::device::Device>,
    size: (u32, u32),
    future: Option<Box<GpuFuture>>,
    ori_state: Arc<ImmutableImage<Format>>,
    tri_state: Arc<ImmutableImage<Format>>,
    tri_extend: Arc<StorageImage<Format>>,
    fb_state: Arc<StorageImage<Format>>,
    back_state: Arc<StorageImage<Format>>,
    fore_state: Arc<StorageImage<Format>>,
    ac_state: Arc<StorageImage<Format>>,
    vertex_buffer: Arc<CpuAccessibleBuffer<[Vertex]>>,
    buf: Arc<CpuAccessibleBuffer<[u8]>>,
}

impl Shared {
    pub fn new<P>(ori_path: P, tri_path: P) -> Self
    where
        P: AsRef<Path>,
    {
        let instance = Instance::new(None, &InstanceExtensions::none(), None).unwrap();
        let physical = PhysicalDevice::enumerate(&instance).next().unwrap();
        // println!( "Using device: {} (type: {:?})", physical.name(), physical.ty() );
        let queue_family = physical.queue_families().find(|&q| q.supports_graphics()).expect("couldn't find a graphical queue family");
        let (device, mut queues) = {
            Device::new(
                physical,
                &Features::none(),
                &DeviceExtensions::none(),
                [(queue_family, 0.5)].iter().cloned(),
            )
            .expect("failed to create device")
        };
        let queue = queues.next().unwrap();
        let vertexs = vec![
            Vertex { position: [-1.0, -1.0] },
            Vertex { position: [ 1.0, -1.0] },
            Vertex { position: [-1.0,  1.0] },
            Vertex { position: [ 1.0,  1.0] },
        ];
        let vertex_buffer = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), vertexs.into_iter()).unwrap();

        let mut matting = Matting::new(ori_path, tri_path);
        matting.run();
        let size = matting.ori_state.dimensions();
        let image = matting.ori_state.into_raw().clone();
        let (width, height) = size;
        let (ori_state, ori_future) = {
            ImmutableImage::from_iter(
                image.iter().cloned(),
                Dimensions::Dim2d {
                    width: width,
                    height: height,
                },
                Format::R8G8B8A8Unorm,
                queue.clone(),
            ).unwrap()
        };
        let buf = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), (0 .. width * height * 4).map(|_| 0u8))
            .expect("failed to create buffer");

        let tri_extend = StorageImage::new(device.clone(), Dimensions::Dim2d { width: width, height: height },
                                Format::R8G8B8A8Unorm, Some(queue.family())).unwrap();
        let fb_state = StorageImage::new(device.clone(), Dimensions::Dim2d { width: width, height: height },
                                Format::R8G8B8A8Unorm, Some(queue.family())).unwrap();
        let back_state = StorageImage::new(device.clone(), Dimensions::Dim2d { width: width, height: height },
                                Format::R8G8B8A8Unorm, Some(queue.family())).unwrap();
        let fore_state = StorageImage::new(device.clone(), Dimensions::Dim2d { width: width, height: height },
                                Format::R8G8B8A8Unorm, Some(queue.family())).unwrap();
        let ac_state = StorageImage::new(device.clone(), Dimensions::Dim2d { width: width, height: height },
                                Format::R8G8B8A8Unorm, Some(queue.family())).unwrap();

        let (tri_state, tri_future) = {
            let image = matting.tri_state.into_raw().clone();
            ImmutableImage::from_iter(
                image.iter().cloned(),
                Dimensions::Dim2d {
                    width: width,
                    height: height,
                },
                Format::R8G8B8A8Unorm,
                queue.clone(),
            )
            .unwrap()
        };

        Self {
            queue: queue,
            device: device,
            size: size,
            future: Some(Box::new(ori_future.join(tri_future)) as Box<GpuFuture>),
            ori_state: ori_state,
            tri_state: tri_state,
            tri_extend: tri_extend,
            fb_state: fb_state,
            back_state: back_state,
            fore_state: fore_state,
            ac_state: ac_state,
            buf: buf,
            vertex_buffer: vertex_buffer,
        }
    }
    
    pub fn run<P>(&mut self, output_path: P)
    where
        P: AsRef<Path>,
    {
        let (width, height) = self.size;

        self.expand_known();
        self.gathering();
        self.refinement();
        self.local_smooth();

        self.future.take().unwrap().then_signal_fence_and_flush().unwrap()
                .wait(None).unwrap();
            
        self.save_image(self.tri_extend.clone(), "0.png");

        let buffer_content = self.buf.read().unwrap();
        let image = ImageBuffer::<Rgba<u8>, _>::from_raw(width, height, &buffer_content[..]).unwrap();
        image.save(output_path).unwrap();
    }

    fn expand_known(&mut self) {
        let tri_state = &self.tri_state;
        let device = &self.device;
        let queue = &self.queue;
        let (width, height) = self.size;
        let ori_state = &self.ori_state;
        let image = &self.tri_extend;
        
        let render_pass = Arc::new(vulkano::single_pass_renderpass!(device.clone(),
            attachments: {
                color: {
                    load: Clear,
                    store: Store,
                    format: Format::R8G8B8A8Unorm,
                    samples: 1,
                }
            },
            pass: {
                color: [color],
                depth_stencil: {}
            }
        ).unwrap());

        let framebuffer = Arc::new(Framebuffer::start(render_pass.clone())
            .add(image.clone()).unwrap()
            .build().unwrap());

        let vs = quad_vs::Shader::load(device.clone()).expect("failed to create shader module");
        let fs = expand_known_fs::Shader::load(device.clone()).expect("failed to create shader module");
        let push_constants = expand_known_fs::ty::PushConstants {
            scale: [width as f32, height as f32],
            kC: 5.0 / 255.0,
            kI: 10.0,
        };

        let pipeline = Arc::new(GraphicsPipeline::start()
            .vertex_input_single_buffer::<Vertex>()
            .vertex_shader(vs.main_entry_point(), ())
            .triangle_strip()
            .viewports_dynamic_scissors_irrelevant(1)
            .fragment_shader(fs.main_entry_point(), ())
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            .build(device.clone())
            .unwrap());

        let dynamic_state = DynamicState {
            viewports: Some(vec![Viewport {
                origin: [0.0, 0.0],
                dimensions: [width as f32, height as f32],
                depth_range: 0.0 .. 1.0,
            }]),
            .. DynamicState::none()
        };

        let sampler = Sampler::new(device.clone(), Filter::Linear, Filter::Linear,
            MipmapMode::Nearest, SamplerAddressMode::Repeat, SamplerAddressMode::Repeat,
            SamplerAddressMode::Repeat, 0.0, 1.0, 0.0, 0.0).unwrap();

        let set = Arc::new(PersistentDescriptorSet::start(pipeline.clone(), 0)
            .add_sampled_image(ori_state.clone(), sampler.clone()).unwrap()
            .add_sampled_image(tri_state.clone(), sampler.clone()).unwrap()
            .build().unwrap()
        );

        let command_buffer = AutoCommandBufferBuilder::primary_one_time_submit(device.clone(), queue.family()).unwrap()
            .begin_render_pass(framebuffer.clone(), false, vec![[0.0, 0.0, 1.0, 1.0].into()]).unwrap()
            .draw(pipeline.clone(), &dynamic_state, self.vertex_buffer.clone(), set.clone(), push_constants).unwrap()
            .end_render_pass().unwrap()
            .build().unwrap();

        self.future = Some(Box::new(self.future.take().unwrap().then_execute(queue.clone(), command_buffer).unwrap()) as Box<GpuFuture>);
    }

    fn gathering(&mut self) {
        let device = &self.device;
        let queue = &self.queue;
        let (width, height) = self.size;
        let ori_state = &self.ori_state;
        let tri_extend = &self.tri_extend;

        let image = &self.fb_state;
        
        let render_pass = Arc::new(vulkano::single_pass_renderpass!(device.clone(),
            attachments: {
                color: {
                    load: Clear,
                    store: Store,
                    format: Format::R8G8B8A8Unorm,
                    samples: 1,
                }
            },
            pass: {
                color: [color],
                depth_stencil: {}
            }
        ).unwrap());

        let framebuffer = Arc::new(Framebuffer::start(render_pass.clone())
            .add(image.clone()).unwrap()
            .build().unwrap());

        let vs = quad_vs::Shader::load(device.clone()).expect("failed to create shader module");
        let fs = gathering_fs::Shader::load(device.clone()).expect("failed to create shader module");
        let push_constants = gathering_fs::ty::PushConstants {
            scale: [width as f32, height as f32],
            isTest: 0,
            kG: 4,
            searchRange: (width+height) as i32,
        };

        let pipeline = Arc::new(GraphicsPipeline::start()
            .vertex_input_single_buffer::<Vertex>()
            .vertex_shader(vs.main_entry_point(), ())
            .triangle_strip()
            .viewports_dynamic_scissors_irrelevant(1)
            .fragment_shader(fs.main_entry_point(), ())
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            .build(device.clone())
            .unwrap());

        let dynamic_state = DynamicState {
            viewports: Some(vec![Viewport {
                origin: [0.0, 0.0],
                dimensions: [width as f32, height as f32],
                depth_range: 0.0 .. 1.0,
            }]),
            .. DynamicState::none()
        };

        let sampler = Sampler::new(device.clone(), Filter::Linear, Filter::Linear,
            MipmapMode::Nearest, SamplerAddressMode::Repeat, SamplerAddressMode::Repeat,
            SamplerAddressMode::Repeat, 0.0, 1.0, 0.0, 0.0).unwrap();

        let set = Arc::new(PersistentDescriptorSet::start(pipeline.clone(), 0)
            .add_sampled_image(ori_state.clone(), sampler.clone()).unwrap()
            .add_sampled_image(tri_extend.clone(), sampler.clone()).unwrap()
            .build().unwrap()
        );

        let command_buffer = AutoCommandBufferBuilder::primary_one_time_submit(device.clone(), queue.family()).unwrap()
            .begin_render_pass(framebuffer.clone(), false, vec![[0.0, 0.0, 1.0, 1.0].into()]).unwrap()
            .draw(pipeline.clone(), &dynamic_state, self.vertex_buffer.clone(), set.clone(), push_constants).unwrap()
            .end_render_pass().unwrap()
            .build().unwrap();

        self.future = Some(Box::new(self.future.take().unwrap().then_execute(queue.clone(), command_buffer).unwrap()) as Box<GpuFuture>);
    }

    fn refinement(&mut self) {
        let device = &self.device;
        let queue = &self.queue;
        let (width, height) = self.size;
        let ori_state = &self.ori_state;
        let tri_extend = &self.tri_extend;
        let fb_state = &self.fb_state;
        let refine_f = &self.fore_state;
        let refine_b = &self.back_state;
        let refine_ac = &self.ac_state;
        
        let render_pass = Arc::new(vulkano::single_pass_renderpass!(device.clone(),
            attachments: {
                color: {
                    load: Clear,
                    store: Store,
                    format: Format::R8G8B8A8Unorm,
                    samples: 1,
                },
                color1: {
                    load: Clear,
                    store: Store,
                    format: Format::R8G8B8A8Unorm,
                    samples: 1,
                },
                color2: {
                    load: Clear,
                    store: Store,
                    format: Format::R8G8B8A8Unorm,
                    samples: 1,
                }
            },
            pass: {
                color: [color, color1, color2],
                depth_stencil: {}
            }
        ).unwrap()); 

        let framebuffer = Arc::new(Framebuffer::start(render_pass.clone())
            .add(refine_f.clone()).unwrap()
            .add(refine_b.clone()).unwrap()
            .add(refine_ac.clone()).unwrap()
            .build().unwrap());

        let vs = quad_vs::Shader::load(device.clone()).expect("failed to create shader module");
        let fs = refinement_fs::Shader::load(device.clone()).expect("failed to create shader module");
        let push_constants = refinement_fs::ty::PushConstants {
            scale: [width as f32, height as f32],
            isTest: 0,
        };

        let pipeline = Arc::new(GraphicsPipeline::start()
            .vertex_input_single_buffer::<Vertex>()
            .vertex_shader(vs.main_entry_point(), ())
            .triangle_strip()
            .viewports_dynamic_scissors_irrelevant(1)
            .fragment_shader(fs.main_entry_point(), ())
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            .build(device.clone())
            .unwrap());

        let dynamic_state = DynamicState {
            viewports: Some(vec![Viewport {
                origin: [0.0, 0.0],
                dimensions: [width as f32, height as f32],
                depth_range: 0.0 .. 1.0,
            }]),
            .. DynamicState::none()
        };

        let sampler = Sampler::new(device.clone(), Filter::Linear, Filter::Linear,
            MipmapMode::Nearest, SamplerAddressMode::Repeat, SamplerAddressMode::Repeat,
            SamplerAddressMode::Repeat, 0.0, 1.0, 0.0, 0.0).unwrap();

        let set = Arc::new(PersistentDescriptorSet::start(pipeline.clone(), 0)
            .add_sampled_image(ori_state.clone(), sampler.clone()).unwrap()
            .add_sampled_image(tri_extend.clone(), sampler.clone()).unwrap()
            .add_sampled_image(fb_state.clone(), sampler.clone()).unwrap()
            .build().unwrap()
        );

        let command_buffer = AutoCommandBufferBuilder::primary_one_time_submit(device.clone(), queue.family()).unwrap()
            .begin_render_pass(framebuffer.clone(), false, vec![
                [0.0, 0.0, 1.0, 1.0].into(),
                [0.0, 0.0, 1.0, 1.0].into(),
                [0.0, 0.0, 1.0, 1.0].into(),
            ]).unwrap()
            .draw(pipeline.clone(), &dynamic_state, self.vertex_buffer.clone(), set.clone(), push_constants).unwrap()
            .end_render_pass().unwrap()
            .build().unwrap();

        self.future = Some(Box::new(self.future.take().unwrap().then_execute(queue.clone(), command_buffer).unwrap()) as Box<GpuFuture>);
    }

    fn local_smooth(&mut self) {
        let device = &self.device;
        let queue = &self.queue;
        let (width, height) = self.size;
        let ori_state = &self.ori_state;
        let tri_extend = &self.tri_extend;
        let back_state = &self.back_state;
        let fore_state = &self.fore_state;
        let ac_state = &self.ac_state;
        let buf = &self.buf;

        let image = StorageImage::new(device.clone(), Dimensions::Dim2d { width: width, height: height },
                                Format::R8G8B8A8Unorm, Some(queue.family())).unwrap();
        
        let render_pass = Arc::new(vulkano::single_pass_renderpass!(device.clone(),
            attachments: {
                color: {
                    load: Clear,
                    store: Store,
                    format: Format::R8G8B8A8Unorm,
                    samples: 1,
                }
            },
            pass: {
                color: [color],
                depth_stencil: {}
            }
        ).unwrap()); 

        let framebuffer = Arc::new(Framebuffer::start(render_pass.clone())
            .add(image.clone()).unwrap()
            .build().unwrap());

        let vs = quad_vs::Shader::load(device.clone()).expect("failed to create shader module");
        let fs = local_smooth_fs::Shader::load(device.clone()).expect("failed to create shader module");
        let push_constants = local_smooth_fs::ty::PushConstants {
            scale: [width as f32, height as f32],
        };

        let pipeline = Arc::new(GraphicsPipeline::start()
            .vertex_input_single_buffer::<Vertex>()
            .vertex_shader(vs.main_entry_point(), ())
            .triangle_strip()
            .viewports_dynamic_scissors_irrelevant(1)
            .fragment_shader(fs.main_entry_point(), ())
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            .build(device.clone())
            .unwrap());

        let dynamic_state = DynamicState {
            viewports: Some(vec![Viewport {
                origin: [0.0, 0.0],
                dimensions: [width as f32, height as f32],
                depth_range: 0.0 .. 1.0,
            }]),
            .. DynamicState::none()
        };

        let sampler = Sampler::new(device.clone(), Filter::Linear, Filter::Linear,
            MipmapMode::Nearest, SamplerAddressMode::Repeat, SamplerAddressMode::Repeat,
            SamplerAddressMode::Repeat, 0.0, 1.0, 0.0, 0.0).unwrap();

        let set = Arc::new(PersistentDescriptorSet::start(pipeline.clone(), 0)
            .add_sampled_image(ori_state.clone(), sampler.clone()).unwrap()
            .add_sampled_image(tri_extend.clone(), sampler.clone()).unwrap()
            .add_sampled_image(fore_state.clone(), sampler.clone()).unwrap()
            .add_sampled_image(back_state.clone(), sampler.clone()).unwrap()
            .add_sampled_image(ac_state.clone(), sampler.clone()).unwrap()
            .build().unwrap()
        );

        let command_buffer = AutoCommandBufferBuilder::primary_one_time_submit(device.clone(), queue.family()).unwrap()
            .begin_render_pass(framebuffer.clone(), false, vec![[0.0, 0.0, 1.0, 1.0].into()]).unwrap()
            .draw(pipeline.clone(), &dynamic_state, self.vertex_buffer.clone(), set.clone(), push_constants).unwrap()
            .end_render_pass().unwrap()
            .copy_image_to_buffer(image.clone(), buf.clone()).unwrap()
            .build().unwrap();

        self.future = Some(Box::new(self.future.take().unwrap().then_execute(queue.clone(), command_buffer).unwrap()) as Box<GpuFuture>);
    }

    fn save_image<P>(&mut self, src_image: Arc<StorageImage<Format>>, path: P)
    where
        P: AsRef<Path>,
    {
        let device = &self.device;
        let queue = &self.queue;
        let (width, height) = self.size;

        let buf = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), (0 .. width * height * 4).map(|_| 0u8))
            .expect("failed to create buffer");
        let image = StorageImage::new(device.clone(), Dimensions::Dim2d { width: width, height: height },
                                Format::R8G8B8A8Unorm, Some(queue.family())).unwrap();
        
        let render_pass = Arc::new(vulkano::single_pass_renderpass!(device.clone(),
            attachments: {
                color: {
                    load: Clear,
                    store: Store,
                    format: Format::R8G8B8A8Unorm,
                    samples: 1,
                }
            },
            pass: {
                color: [color],
                depth_stencil: {}
            }
        ).unwrap()); 

        let framebuffer = Arc::new(Framebuffer::start(render_pass.clone())
            .add(image.clone()).unwrap()
            .build().unwrap());

        let vs = quad_vs::Shader::load(device.clone()).expect("failed to create shader module");
        let fs = copy_fs::Shader::load(device.clone()).expect("failed to create shader module");
        let push_constants = local_smooth_fs::ty::PushConstants {
            scale: [width as f32, height as f32],
        };

        let pipeline = Arc::new(GraphicsPipeline::start()
            .vertex_input_single_buffer::<Vertex>()
            .vertex_shader(vs.main_entry_point(), ())
            .triangle_strip()
            .viewports_dynamic_scissors_irrelevant(1)
            .fragment_shader(fs.main_entry_point(), ())
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            .build(device.clone())
            .unwrap());

        let dynamic_state = DynamicState {
            viewports: Some(vec![Viewport {
                origin: [0.0, 0.0],
                dimensions: [width as f32, height as f32],
                depth_range: 0.0 .. 1.0,
            }]),
            .. DynamicState::none()
        };

        let sampler = Sampler::new(device.clone(), Filter::Linear, Filter::Linear,
            MipmapMode::Nearest, SamplerAddressMode::Repeat, SamplerAddressMode::Repeat,
            SamplerAddressMode::Repeat, 0.0, 1.0, 0.0, 0.0).unwrap();

        let set = Arc::new(PersistentDescriptorSet::start(pipeline.clone(), 0)
            .add_sampled_image(src_image.clone(), sampler.clone()).unwrap()
            .build().unwrap()
        );

        let command_buffer = AutoCommandBufferBuilder::primary_one_time_submit(device.clone(), queue.family()).unwrap()
            .begin_render_pass(framebuffer.clone(), false, vec![[0.0, 0.0, 1.0, 1.0].into()]).unwrap()
            .draw(pipeline.clone(), &dynamic_state, self.vertex_buffer.clone(), set.clone(), push_constants).unwrap()
            .end_render_pass().unwrap()
            .copy_image_to_buffer(image.clone(), buf.clone()).unwrap()
            .build().unwrap();

        let finished = command_buffer.execute(queue.clone()).unwrap();
        finished.then_signal_fence_and_flush().unwrap().wait(None).unwrap();

        let buffer_content = buf.read().unwrap();
        let image = ImageBuffer::<Rgba<u8>, _>::from_raw(width, height, &buffer_content[..]).unwrap();
        image.save(path).unwrap();
    }
}

#[inline]
fn dc(a: image::Rgba<u8>, b: image::Rgba<u8>) -> f32 {
    return ((a[0] as i32 - b[0] as i32).abs()
        + (a[1] as i32 - b[1] as i32).abs()
        + (a[2] as i32 - b[2] as i32).abs()) as f32
        / 765.0;
}

#[derive(Debug)]
pub struct Matting {
    pub ori_state: ImageBuffer<image::Rgba<u8>, Vec<u8>>,
    pub tri_state: ImageBuffer<image::Rgba<u8>, std::vec::Vec<u8>>,
    tables: Vec<Vec<Vec<(f32, f32)>>>,
    stacks: Vec<LinkedList<(u32, u32, f32)>>,
    width: u32,
    height: u32,
}

impl Matting {
    pub fn new<P>(ori_path: P, tri_path: P) -> Self
    where
        P: AsRef<Path>,
    {
        let ori = image::open(ori_path).unwrap();
        let tri = image::open(tri_path).unwrap();
        let ori_state = ori.to_rgba();
        let tri_state = tri.to_rgba();
        return Self::from(ori_state, tri_state);
    }

    pub fn from(
        ori_state: ImageBuffer<image::Rgba<u8>, Vec<u8>>,
        tri_state: ImageBuffer<image::Rgba<u8>, std::vec::Vec<u8>>,
    ) -> Self {
        let (width, height) = tri_state.dimensions();
        let tables = vec![vec![vec![(0.0, 0.0); height as usize]; width as usize]; 2];
        let stacks: Vec<LinkedList<(u32, u32, f32)>> = vec![LinkedList::new(); 2];
        Self {
            ori_state: ori_state,
            tri_state: tri_state,
            stacks: stacks,
            tables: tables,
            width: width,
            height: height,
        }
    }

    fn compare(&mut self, (x, y): (u32, u32), cur: f32, color: image::Rgba<u8>, idx: usize) {
        if x < self.width && y < self.height {
            let (d, _) = self.tables[idx][x as usize][y as usize];
            if d < 0.98 {
                // 相似度已经很高了
                let neighbor_color = self.ori_state.get_pixel(x, y);
                // 相邻相似度
                let diff = 1.0 - dc(color, *neighbor_color);
                // 相对相似度
                if diff > 0.98 {
                    let val = diff * cur;
                    self.tables[idx][x as usize][y as usize] = (diff, val);
                    if diff > 0.98 {
                        self.stacks[idx].push_back((x, y, val));
                    }
                }
            }
        }
    }

    fn expand(&mut self, x: u32, y: u32, v: f32, idx: usize) {
        let color = *self.ori_state.get_pixel(x, y);
        if x > 0 {
            if y > 0 {
                self.compare((x - 1, y - 1), v, color, idx);
            }
            self.compare((x - 1, y), v, color, idx);
            if y < self.height - 1 {
                self.compare((x - 1, y + 1), v, color, idx);
            }
        }
        if y > 0 {
            self.compare((x, y - 1), v, color, idx);
        }
        self.compare((x, y), v, color, idx);
        if y < self.height - 1 {
            self.compare((x, y + 1), v, color, idx);
        }
        if x < self.width - 1 {
            if y > 0 {
                self.compare((x + 1, y - 1), v, color, idx);
            }
            self.compare((x + 1, y), v, color, idx);
            if y < self.height - 1 {
                self.compare((x + 1, y + 1), v, color, idx);
            }
        }
    }

    pub fn run(&mut self) {
        for x in 0..self.width {
            for y in 0..self.height {
                let pixel = self.tri_state.get_pixel(x, y);
                if pixel[0] == 255 {
                    // 前景
                    self.tables[0][x as usize][y as usize] = (1.0, 1.0);
                    self.stacks[0].push_back((x, y, 1.0));
                } else if pixel[0] == 0 {
                    // 背景
                    self.tables[1][x as usize][y as usize] = (1.0, 1.0);
                    self.stacks[1].push_back((x, y, 1.0));
                }
            }
        }
        for idx in 0..2 {
            while self.stacks[idx].len() > 0 {
                if let Some((x, y, v)) = self.stacks[idx].pop_front() {
                    self.expand(x, y, v, idx);
                }
            }
        }
        for x in 0..self.width {
            for y in 0..self.height {
                let pixel = self.tri_state.get_pixel(x, y);
                if pixel[0] > 0 && pixel[0] < 255 {
                    let (fd, fv) = self.tables[0][x as usize][y as usize];
                    let (bd, bv) = self.tables[1][x as usize][y as usize];
                    if fd >= 0.98 && fv > bv {
                        self.tri_state.put_pixel(x, y, image::Rgba([255,255,255,255]));
                    }
                    if bd >= 0.98 && bv > fv {
                        self.tri_state.put_pixel(x, y, image::Rgba([0,0,0,255]));
                    }
                }
            }
        }
    }
}

#[no_mangle]
pub extern "C" fn run(ori_path: *const c_char, tri_path: *const c_char, output_path: *const c_char) {
	let ori_path = unsafe {
		assert!(!ori_path.is_null());
		CStr::from_ptr(ori_path)
	};
	let ori_path = ori_path.to_str().unwrap();
	let tri_path = unsafe {
		assert!(!tri_path.is_null());
		CStr::from_ptr(tri_path)
	};
	let tri_path = tri_path.to_str().unwrap();
	let output_path = unsafe {
		assert!(!output_path.is_null());
		CStr::from_ptr(output_path)
	};
	let output_path = output_path.to_str().unwrap();
	let mut shared = Shared::new(ori_path, tri_path);
	shared.run(output_path);
}
